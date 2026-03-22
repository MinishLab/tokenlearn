---
{{ card_data }}
---

# {{ repo_id or dataset_name }} Dataset Card

This dataset was created with [Tokenlearn](https://github.com/MinishLab/tokenlearn) for training [Model2Vec](https://github.com/MinishLab/model2vec) models. It contains mean token embeddings produced by a sentence transformer, used as training targets for static embedding distillation.

## Dataset Details

| Field | Value |
|---|---|
| **Source dataset** | [{{ source_dataset }}](https://huggingface.co/datasets/{{ source_dataset }}) |
| **Source split** | `{{ source_split }}` |
| **Embedding model** | [{{ model_name }}](https://huggingface.co/{{ model_name }}) |
| **Embedding dimension** | {{ embedding_dim }} |
| **Rows** | {{ num_rows }} |

## Dataset Structure

| Column | Type | Description |
|---|---|---|
| `text` | `string` | Truncated input text |
| `embedding` | `list[float32]` | Mean token embedding from `{{ model_name }}`, excluding BOS/EOS tokens |

## Usage

Load with the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("{{ repo_id or dataset_name }}")
```

Train a Model2Vec model on this dataset using Tokenlearn:

```bash
python -m tokenlearn.train \
    --model-name "{{ model_name }}" \
    --data-path "{{ repo_id or dataset_name }}" \
    --save-path "<path-to-save-model>"
```

## Creation

This dataset was created using the `tokenlearn-featurize` CLI:

```bash
python -m tokenlearn.featurize \
    --model-name "{{ model_name }}" \
    --dataset-path "{{ source_dataset }}" \
    --dataset-name "{{ source_name }}" \
    --dataset-split "{{ source_split }}" \
    --output-dir "<output-dir>"
```

## Library Authors

Tokenlearn was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of [Stephan Tulkens](https://github.com/stephantul) and [Thomas van Dongen](https://github.com/Pringled).

## Citation

```
@article{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}
```
