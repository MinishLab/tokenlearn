import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluation import CustomMTEB, TaskType, get_tasks, make_leaderboard, parse_mteb_results, summarize_results
from model2vec import StaticModel
from model2vec.distill import distill
from model2vec.distill.distillation import _post_process_embeddings
from reach import Reach
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nanofit.train import TextDataset, train_supervised


def collect_means_and_texts(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a bunch of reach paths."""
    txts = []
    v = []
    for path in paths:
        if not path.name.endswith(".json"):
            continue
        try:
            r = Reach.load(path)
        except KeyError:
            # Workaround for old format reach
            vectors_path = str(path).replace("_items.json", "_vectors.npy")
            items = json.load(open(path))["items"]
            vectors = np.load(open(vectors_path, "rb"))
            r = Reach(vectors, items)
        # Filter out any NaN vectors before appending
        non_nan_indices = ~np.isnan(r.vectors).any(axis=1)
        valid_vectors = r.vectors[non_nan_indices]
        valid_items = np.array(r.sorted_items)[non_nan_indices]
        txts.extend(valid_items)
        v.append(valid_vectors)

    return txts, np.concatenate(v)


def train_model(args: Any) -> StaticModel:
    """Main function to train a StaticModel."""
    logging.basicConfig(level=logging.INFO)

    # Use distill or random embeddings based on CLI args
    if args.random_embeddings:
        logging.info("Using random embeddings.")
        s = distill(args.model_name)
        v = np.random.randn(*s.embedding.shape)  # noqa NPY002
        v = _post_process_embeddings(v, "auto", False).astype(np.float32)
        s = StaticModel(v, s.tokenizer)
    else:
        s = distill(args.model_name)

    # Collect paths for training
    paths = sorted(Path(args.data_path).glob("*.json"))
    train_txt, train_vec = collect_means_and_texts(paths)
    train_data = TextDataset(train_txt, torch.from_numpy(train_vec), s.tokenizer)

    # Train the model
    model, _ = train_supervised(train_data, s, device=args.device)

    # Save the trained model
    model.save_pretrained(args.save_path)

    return model


def test_model(args: Any, model_name: str) -> None:
    """Function to test the StaticModel."""
    # Get all available tasks for evaluation
    tasks = get_tasks(
        ["Classification", "Clustering", "PairClassification", "Reranking", "STS", "Summarization", "WordSim", "PEARL"]
    )

    evaluation = CustomMTEB(tasks=tasks)

    # Load the trained model
    model = StaticModel.from_pretrained(model_name)

    logging.info("Applying reweighting and PCA to the model.")
    paths = sorted(Path(args.data_path).glob("*.json"))
    txt, _ = collect_means_and_texts(paths)

    counts: Counter[str] = Counter()
    for t in tqdm(txt):
        counts.update(model.tokenizer.encode(t, add_special_tokens=False).ids)

    sum_id = sum(counts.values()) + len(model.tokens)
    x = np.full(len(model.embedding), 1 / sum_id)

    for word_id, count in counts.items():
        x[word_id] = (count + 1) / sum_id

    w = model.embedding
    w = np.nan_to_num(w)

    dim = 256
    p = PCA(n_components=dim)
    w = p.fit_transform(w)

    alpha = 1e-3
    f = alpha / (alpha + x)
    w *= f[:, None]
    model.embedding = w
    model.normalize = True

    model.save_pretrained(f"{model_name}_weighted")

    # Run the evaluation
    logging.info("Running the evaluation.")
    results = evaluation.run(model, eval_splits=["test"], output_folder=f"results", overwrite_results=True)

    # Parse and summarize results
    parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
    task_scores = summarize_results(parsed_results)

    # Print results in leaderboard format
    leaderboard = make_leaderboard(task_scores)
    for idx, row in leaderboard.iterrows():
        print(row)  # noqa T201


def main(args: Any) -> None:
    """Main function to handle both training and testing."""
    # Train the model
    model = train_model(args)

    # Test the model if requested
    test_model(args, args.save_path)


if __name__ == "__main__":
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Train and Test StaticModel with distillation and dataset paths.")

    # Training arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/fineweb_bgebase", help="Path to the directory containing the dataset."
    )
    parser.add_argument("--save-path", type=str, help="Path to save the trained model.")

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the training on (e.g., 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--random-embeddings", action="store_true", help="Use random embeddings instead of distilling the model."
    )

    args = parser.parse_args()

    # Run main function
    main(args)
