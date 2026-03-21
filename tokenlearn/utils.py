import logging
from collections import Counter
from pathlib import Path

import numpy as np
import regex
from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_vocab(texts: list[str], vocab_size: int = 56_000) -> list[str]:
    """
    Create a vocabulary from a list of texts.

    :param texts: The list of texts to create the vocabulary from.
    :param vocab_size: The size of the vocabulary. Defaults to 56,000, which is the vocab_size used for our 32M models.
    :return: The vocabulary.
    """
    tokenizer_regex = regex.compile(r"\w+|[^\w\s]+")

    # Tokenize all texts
    tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        tokens.extend(tokenizer_regex.findall(text.lower()))

    # Count the tokens
    token_counts = Counter(tokens)

    # Get the most common tokens as the vocabulary
    vocab = [word for word, _ in token_counts.most_common(vocab_size)]
    return vocab


def collect_means_and_texts(data_path: str | Path, max_samples: int | None = None) -> tuple[list[str], np.ndarray]:
    """Collect means and texts from a HuggingFace dataset directory."""
    shards = []
    rows = 0
    for shard_dir in sorted(Path(data_path).glob("shard_*/")):
        shard = load_from_disk(str(shard_dir))
        shards.append(shard)
        rows += len(shard)
        if max_samples and rows >= max_samples:
            break

    dataset = concatenate_datasets(shards)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    texts = dataset["text"]
    vectors = np.array(dataset["embedding"], dtype=np.float32)

    non_nan_mask = ~np.isnan(vectors).any(axis=1)
    texts = np.array(texts)[non_nan_mask].tolist()
    vectors = vectors[non_nan_mask]

    if max_samples:
        texts = texts[:max_samples]
        vectors = vectors[:max_samples]

    return texts, vectors
