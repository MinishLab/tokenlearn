from evaluation import CustomMTEB, TaskType, get_tasks, make_leaderboard, parse_mteb_results, summarize_results
from evaluation.classification_benchmark import ClassificationBenchmark
from model2vec import StaticModel
from mteb import ModelMeta

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
    ]
)
# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)

# Load the model
model_name = "potion_w_mse_loss_256_norm"
model = StaticModel.from_pretrained(model_name)

# Optionally, add model metadata in MTEB format
model.mteb_model_meta = ModelMeta(name=model_name, revision="no_revision_available", release_date=None, languages=None)

# Run the evaluation
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results")

# Parse the results and summarize them
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)

# Print the results in a leaderboard format
leaderboard = make_leaderboard(task_scores)
