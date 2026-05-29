import logging
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import regex
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_vocab(texts: list[str], vocab_size: int = 56_000) -> list[str]:
    """Create a vocabulary from a list of texts.

    :param texts: The list of texts to create the vocabulary from.
    :param vocab_size: The size of the vocabulary. Defaults to 56,000, which is the vocab_size used for our 32M models.
    :return: The vocabulary.
    """
    tokenizer_regex = regex.compile(r"\w+|[^\w\s]+")

    # Tokenize all texts
    token_counts: Counter[str] = Counter()
    for text in tqdm(texts, desc="Tokenizing texts"):
        for word in tokenizer_regex.findall(text.lower()):
            token_counts[word] += 1

    # Get the most common tokens as the vocabulary
    vocab = [word for word, _ in token_counts.most_common(vocab_size)]
    return vocab


def collect_means_and_texts(
    data_path: str | Path,
    max_samples: int | None = None,
    split: str | None = "train",
    name: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a local HuggingFace dataset directory or Hub repo."""
    if Path(data_path).exists():
        # If path exists, load from disk
        dataset = load_from_disk(str(data_path))
        if split:
            dataset = dataset[split]
    else:
        # Attempt to load from HF hub
        dataset = load_dataset(str(data_path), name=name, split=split)
    # We know that it will be a dataset here.
    dataset = cast(Dataset, dataset)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    texts = dataset["text"]
    vectors = np.array(dataset["embedding"], dtype=np.float32)

    # Filter out any rows where the vector contains NaN values
    non_nan_mask = ~np.isnan(vectors).any(axis=1)
    texts = np.array(texts)[non_nan_mask].tolist()
    vectors = vectors[non_nan_mask]

    if max_samples:
        texts = texts[:max_samples]
        vectors = vectors[:max_samples]

    return texts, vectors
