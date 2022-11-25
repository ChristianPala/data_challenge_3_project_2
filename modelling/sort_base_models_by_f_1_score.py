# Auxiliary library to sort all the base models in the results sub-folders by their f1 score and print
# the best overall model to the console.

# Libraries:
# Data manipulation:
from pathlib import Path
import shutil

# Global variables:
results_path: Path = Path("..", "results")
trees_results_path: Path = Path(results_path, "trees")
neural_networks_results_path: Path = Path(results_path, "neural_network")
other_models_results_path: Path = Path(results_path, "other_models")


# Functions:
def sort_results_by_f_1_score(path: Path) -> Path:
    """
    This function sorts all the models in the results subfolders by their f1 score.
    @param path: Path: the path to the results' subfolder.
    :return: None.
    """
    # Get all the models' results:
    models_results = list(path.glob("*.txt"))
    # txt files are ordered by key, value pairs, f1 score is the first value.
    # Sort the models' results by their f1 score:
    models_results.sort(key=lambda x: float(x.read_text().split(",")[0].split(":")[1].split("\n")[0]))
    # store the best model in the best_models subfolder:
    best_models_path: Path = Path(path, "best_models")
    best_models_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(models_results[-1], best_models_path)
    # return the name of the best model:
    return models_results[-1]


def get_best_overall_model(tree: Path, neural_network: Path, other_model: Path) -> Path:
    """
    This function gets the best overall model.
    @param tree: str: the name of the best tree model.
    @param neural_network: str: the name of the best neural network model.
    @param other_model: str: the name of the best other model.
    :return: str: the name of the best overall model.
    """
    # get the f1 scores:
    tree_f1_score: float = float(tree.read_text().split(",")[0].split(":")[1].split("\n")[0])
    neural_network_f1_score: float = float(neural_network.read_text().split(",")[0].split(":")[1].split("\n")[0])
    other_model_f1_score: float = float(other_model.read_text().split(",")[0].split(":")[1].split("\n")[0])
    # get the best model:
    best_model: Path = tree if tree_f1_score > neural_network_f1_score and tree_f1_score > other_model_f1_score else \
        neural_network if neural_network_f1_score > other_model_f1_score else other_model

    return best_model


def sort_all_results_by_f_1_score(suppress_print=False) -> None:
    """
    This function sorts all the models in the results subfolders by their f1 score.
    :return: None.
    """
    tree: Path = sort_results_by_f_1_score(trees_results_path)
    nn: Path = sort_results_by_f_1_score(neural_networks_results_path)
    other: Path = sort_results_by_f_1_score(other_models_results_path)

    tree_score: float = float(tree.read_text().split(",")[0].split(":")[1].split("\n")[0])
    nn_score: float = float(nn.read_text().split(",")[0].split(":")[1].split("\n")[0])
    other_score: float = float(other.read_text().split(",")[0].split(":")[1].split("\n")[0])

    best = get_best_overall_model(tree, nn, other)
    best_score = best.read_text().split(",")[0].split(":")[1].split("\n")[0]
    model_description: str = best.name.split('.')[0].replace('_', ' ') \
        .replace('base evaluation results project 2 dataset ', 'with ')
    model_description = model_description.rsplit(' ', 1)[0] + ", imputation strategy: " + \
                        model_description.rsplit(' ', 1)[1]

    if not suppress_print:
        print(f"The best overall base model is {model_description}, with an f1 score of {round(float(best_score), 3)}")


if __name__ == "__main__":
    sort_all_results_by_f_1_score()
