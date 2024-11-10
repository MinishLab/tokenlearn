import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from more_itertools import batched
from tokenizers import Tokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_means_and_texts(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a list of paths."""
    txts = []
    vectors_list = []
    for items_path in tqdm(paths, desc="Collecting means and texts"):
        if not items_path.name.endswith("_items.json"):
            continue
        base_path = items_path.with_name(items_path.stem.replace("_items", ""))
        vectors_path = base_path.with_name(base_path.name + "_vectors.npy")
        try:
            with open(items_path, "r") as f:
                data = json.load(f)
            items = data.get("items", [])
            vectors = np.load(vectors_path, allow_pickle=False)
        except (KeyError, FileNotFoundError, ValueError) as e:
            logger.info(f"Error loading data from {base_path}: {e}")
            continue

        # Filter out any NaN vectors before appending
        vectors = np.array(vectors)
        items = np.array(items)
        non_nan_indices = ~np.isnan(vectors).any(axis=1)
        valid_vectors = vectors[non_nan_indices]
        valid_items = items[non_nan_indices]
        txts.extend(valid_items.tolist())
        vectors_list.append(valid_vectors)

    if vectors_list:
        all_vectors = np.concatenate(vectors_list, axis=0)
    else:
        all_vectors = np.array([])
    return txts, all_vectors


def calculate_token_probabilities(tokenizer: Tokenizer, txt: list[str]) -> np.ndarray:
    """Count tokens in a set of texts."""
    vocab_size = tokenizer.get_vocab_size()
    counts: Counter[int] = Counter()
    for t in tqdm(batched(txt, 1024)):
        encodings = tokenizer.encode_batch_fast(t, add_special_tokens=False)
        for e in encodings:
            counts.update(e.ids)

    # Add the number of tokens to account for smoothing
    sum_id = sum(counts.values()) + vocab_size
    # Start with ones for smoothing
    x = np.ones(vocab_size)

    for word_id, count in counts.items():
        x[word_id] += count

    # Turn into probabilities
    x /= sum_id

    return x
