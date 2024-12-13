# Tokenlearn
Tokenlearn is a method to pre-train [Model2Vec](https://github.com/MinishLab/model2vec).

The method is described in detail in our [Tokenlearn blogpost](https://minishlab.github.io/tokenlearn_blogpost/).

## Quickstart

Install the package with:

```bash
pip install tokenlearn
```

The basic usage of Tokenlearn consists of two CLI scripts: `featurize` and `train`.

Tokenlearn is trained using means from a sentence transformer. To create means, the `tokenlearn-featurize` CLI can be used:

```bash
python3 -m tokenlearn.featurize --model-name "baai/bge-base-en-v1.5" --output-dir "data/c4_features"
```

NOTE: the default model is trained on the C4 dataset. If you want to use a different dataset, the following code can be used:

```bash
python3 -m tokenlearn.featurize \
    --model-name "baai/bge-base-en-v1.5" \
    --output-dir "data/c4_features" \
    --dataset-path "allenai/c4" \
    --dataset-name "en" \
    --dataset-split "train"
```

To train a model on the featurized data, the `tokenlearn-train` CLI can be used:
```bash
python3 -m tokenlearn.train --model-name "baai/bge-base-en-v1.5" --data-path "data/c4_features" --save-path "<path-to-save-model>"
```

Training will create two models:
- The base trained model.
- The base model with weighting applied. This is the model that should be used for downstream tasks.

NOTE: the code assumes that the padding token ID in your tokenizer is 0. If this is not the case, you will need to modify the code.

### Evaluation

To evaluate a model, you can use the following command after installing the optional evaluation dependencies:

```bash
pip install evaluation@git+https://github.com/MinishLab/evaluation@main

```

```python
from model2vec import StaticModel

from evaluation import CustomMTEB, get_tasks, parse_mteb_results, make_leaderboard, summarize_results
from mteb import ModelMeta

# Get all available tasks
tasks = get_tasks()
# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)

# Load a trained model
model_name = "tokenlearn_model"
model = StaticModel.from_pretrained(model_name)

# Optionally, add model metadata in MTEB format
model.mteb_model_meta = ModelMeta(
            name=model_name, revision="no_revision_available", release_date=None, languages=None
        )

# Run the evaluation
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results")

# Parse the results and summarize them
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)

# Print the results in a leaderboard format
print(make_leaderboard(task_scores))
```

## License

MIT
