
<h2 align="center">
  <img width="25%" alt="Tokenlearn logo" src="assets/images/tokenlearn_logo.webp"><br/>
  Pre-train Static Word Embeddings
</h2>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/tokenlearn/"><img src="https://img.shields.io/pypi/v/tokenlearn?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://minish.ai/packages/tokenlearn/usage/"><img src="https://img.shields.io/badge/docs-minish.ai-blue.svg" alt="Docs"></a>
    <a href="https://github.com/MinishLab/tokenlearn/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
  </h2>

[Quickstart](#quickstart) •
[Featurize](#featurize) •
[Train](#train) •
[Evaluation](#evaluation)

</div>

Tokenlearn is a method to pre-train [Model2Vec](https://github.com/MinishLab/model2vec) static embedding models. The original version, used to train the [potion](https://huggingface.co/collections/minishlab/potion-6721e9e87bc4404a85e3467b) models (potion-base-2M/4M/8M/32M), is described in our [original blogpost](https://minish.ai/blog/2024-10-29-tokenlearn-blogpost/). The current version, used to train the multilingual potion models, is covered in our [Tokenlearn 2.0 post](https://minish.ai/blog/2025-05-31-tokenlearn-release/).

## Quickstart

Install the package with:

```bash
pip install tokenlearn
```

Tokenlearn consists of two steps: **featurize** (create mean token embeddings from a sentence transformer) and **train** (pre-train a static Model2Vec model using those embeddings as targets).

## Featurize

Use the `tokenlearn.featurize` CLI to create a featurized dataset from any HuggingFace dataset:

```bash
python -m tokenlearn.featurize \
    --model-name "baai/bge-base-en-v1.5" \
    --dataset-path "allenai/c4" \
    --dataset-name "en" \
    --dataset-split "train" \
    --output-dir "data/c4_features"
```

The output is a standard HuggingFace dataset saved to `--output-dir`. You can optionally push it to the Hub after featurizing:

```bash
python -m tokenlearn.featurize \
    --model-name "baai/bge-base-en-v1.5" \
    --output-dir "data/c4_features" \
    --push-to-hub "username/my-featurized-dataset"
```

## Train

Use the `tokenlearn.train` CLI to train a Model2Vec model on a featurized dataset:

```bash
python -m tokenlearn.train \
    --model-name "baai/bge-base-en-v1.5" \
    --data-path "data/c4_features" \
    --save-path "<path-to-save-model>"
```

`--data-path` also accepts a HuggingFace Hub repo ID:

```bash
python -m tokenlearn.train \
    --model-name "baai/bge-base-en-v1.5" \
    --data-path "username/my-featurized-dataset" \
    --save-path "<path-to-save-model>"
```

Training produces two models:
- The base trained model.
- The base model with weighting applied — this is the model to use for downstream tasks.

> **Note**: the code assumes the padding token ID in your tokenizer is 0. If this is not the case, you will need to modify the code.

## Evaluation

To evaluate a trained model, install the optional evaluation dependencies:

```bash
pip install evaluation@git+https://github.com/MinishLab/evaluation@main
```

<details>
<summary>Show evaluation code</summary>
<br>

```python
from model2vec import StaticModel

from evaluation import CustomMTEB, get_tasks, parse_mteb_results, make_leaderboard, summarize_results
from mteb import ModelMeta

# Get all available tasks
tasks = get_tasks()
evaluation = CustomMTEB(tasks=tasks)

# Load a trained model
model_name = "tokenlearn_model"
model = StaticModel.from_pretrained(model_name)

# Optionally, add model metadata in MTEB format
model.mteb_model_meta = ModelMeta(
    name=model_name, revision="no_revision_available", release_date=None, languages=None
)

# Run the evaluation
results = evaluation.run(model, eval_splits=["test"], output_folder="results")

# Parse and print results
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)
print(make_leaderboard(task_scores))
```

</details>

## License

MIT

## Citing

If you use Tokenlearn in your research, please cite the following:

```bibtex
@software{minishlab2024model2vec,
  author       = {Stephan Tulkens and {van Dongen}, Thomas},
  title        = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17270888},
  url          = {https://github.com/MinishLab/model2vec},
  license      = {MIT}
}
```
