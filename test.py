from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from evaluation import CustomMTEB, TaskType, get_tasks, make_leaderboard, parse_mteb_results, summarize_results
from evaluation.classification_benchmark import ClassificationBenchmark
from model2vec import StaticModel
from mteb import ModelMeta
from reach import Reach
from sklearn.decomposition import PCA
from tqdm import tqdm


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
            # whatever
            vectors_path = str(path).replace("_items.json", "_vectors.npy")
            items = json.load(open(path))["items"]
            vectors = np.load(open(vectors_path, "rb"))
            r = Reach(vectors, items)
        txts.extend(r.sorted_items)
        v.append(r.vectors)

    return txts, np.concatenate(v)


# Get all available tasks
tasks = get_tasks(
    [
        TaskType.STS,
        TaskType.WORDSIM,
        TaskType.PEARL,
        TaskType.PAIRCLASSIFICATION,
        TaskType.CLASSIFICATION,
        TaskType.CLUSTERING,
        TaskType.RERANKING,
        TaskType.SUMMARIZATION,
        # TaskType.RETRIEVAL,
    ]
)
# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)

# Load the model
model_name = "potion_bge_base_768_new"
model = StaticModel.from_pretrained(model_name)
dim = model.dim

# NOTE: uncomment this if your model is not reweighted yet.
# This code block will reweight the model's embeddings based on the
# counts of the words in the training data.
# And then applies PCA.
paths = sorted(Path("data/c4_old").glob("*.json"))
paths.extend(sorted(Path("data/fineweb").glob("*.json")))

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

model.embedding = w

suffix = f"_reweight+pca_{dim}+fineweb_counts_NORMalized_ccc"
full_model_name = model_name + suffix

# Optionally, add model metadata in MTEB format
model.mteb_model_meta = ModelMeta(
    name=full_model_name,
    revision="no_revision_available",
    release_date=None,
    languages=None,
)

# Run the evaluation
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results")

# Parse the results and summarize them
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)

# Print the results in a leaderboard format
leaderboard = make_leaderboard(task_scores)

clf = ClassificationBenchmark(model, f"results/{full_model_name}")
clf.run()
