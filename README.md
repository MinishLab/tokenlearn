# Tokenlearn
Tokenlearn is a method to pre-train [Model2Vec](https://github.com/MinishLab/model2vec).

The method is described in detail in our [Tokenlearn blogpost](https://minishlab.github.io/tokenlearn_blogpost/).

## Usage

### Featurizing
Tokenlearn is trained using means from a sentence transformer. To create means, the `featurize` script can be used:

```bash
python tokenlearn/featurize.py
```

This will create means for [C4](https://huggingface.co/datasets/allenai/c4) using [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5).

### Training
The easiest way to train using Tokenlearn is to use the CLI. You can use the following command to train a model:

```bash
python train.py --data-path <path-to-your-data> --save-path <path-to-save-model>
```

Training will create two models:
- The base trained model.
- The base model with weighting applied. This is the model that should be used for downstream tasks.

NOTE: the code assumes that the padding token ID in your tokenizer is 0. If this is not the case, you will need to modify the code.

### Evaluating

To evaluate a model, you can use the following command:

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
