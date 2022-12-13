# Auxiliary library to save the model results to a csv file.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
# Metrics:
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    classification_report, confusion_matrix
# Global variables:
from config import trees_results_path, neural_networks_results_path, other_models_results_path, results_path, \
    neural_networks_balanced_results_path, trees_balanced_results_path, other_models_balanced_results_path


# Functions:
def evaluate_model(y_test: np.array, y_pred: np.array) -> dict[str, float]:
    """
    This function evaluates the model's performance.
    @param y_test: np.array: the target values for the test data.
    @param y_pred: np.array: the predicted target values.
    :return: Dictionary with the metrics.
    """
    y_pred = np.where(y_pred > 0.5, 1, 0)
    df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

    return {
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'confusion_matrix': np.array2string(confusion_matrix(y_test, y_pred, normalize='true'), precision=2,
                                            separator=', '),
        'classification_report': df
    }


def save_evaluation_results(evaluation_results: dict, model_type: str, save_path: Path,
                            dataset_name: str) -> None:
    """
    This function saves the evaluation results of the various models to a .txt file.
    @param evaluation_results: dict: the dictionary with the evaluation results.
    @param model_type: str: the type of the model.
    @param save_path: Path: the path to save the results to.
    @param dataset_name: str: the name of the dataset to save the results for.
    :return: None. Saves the results to a file in the results' folder.
    """
    # ensure the path exists:
    save_path.mkdir(parents=True, exist_ok=True)

    # write the results to a file:
    with open(save_path / f'{model_type}_{dataset_name}.txt', 'w') as f:
        for key, value in evaluation_results.items():
            if key == 'confusion_matrix':
                f.write(f'{key}\n {value}\n')
            elif key == 'classification_report':
                # empty line
                f.write('\n')
                value.to_csv(f, mode='a', header=True, sep='\t')
            else:
                f.write(f'{key}: {value}\n')


def sort_results_by_f_1_score(path: Path) -> Path:
    """
    This function sorts all the models in the results subfolders by their f1 score.
    @param path: Path: the path to the results' subfolder.
    :return: None.
    """
    # Get all the models' results:
    models_results = list(path.rglob("*.txt"))
    # txt files are ordered by key, value pairs, f1 score is the first value.
    # Sort the models' results by their f1 score:
    models_results.sort(key=lambda x: float(x.read_text().split(",")[0].split(":")[1].split("\n")[0]))
    # store the best model in the best_models sub-folder:
    best_models_path: Path = Path(path, "best_models")
    best_models_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(models_results[-1], best_models_path)
    # return the name of the best model:
    return models_results[-1]


def get_best_overall_model(tree: Path, neural_network: Path, other_model: Path) -> Path:
    """
    This function gets the best overall model, assumes the paths are sorted by f1 score.
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


def sort_all_results_by_f_1_score(tree_path: Path, neural_network_path: Path, other_models_path: Path,
                                  suppress_print=False) -> None:
    """
    This function sorts all the models in the results subfolders by their f1 score.
    :return: None.
    """

    tree: Path = sort_results_by_f_1_score(tree_path)
    nn: Path = sort_results_by_f_1_score(neural_network_path)
    other: Path = sort_results_by_f_1_score(other_models_path)

    best = get_best_overall_model(tree, nn, other)

    best_score = best.read_text().split(",")[0].split(":")[1].split("\n")[0]
    model_description: str = best.name.split('.')[0].replace('_', ' ') \
        .replace('base evaluation results project 2 dataset ', '')

    if not suppress_print:
        print(f"The best overall base model is {model_description}, with an f1 score of {round(float(best_score), 3)}")


def get_best_model_by_type(model_type: str, path: Path) -> Path:
    """
    This function gets the best model of a specific type.
    @param model_type: str: the type of the model.
    @param path: Path: the path to the results' subfolder.
    :return: Path: the path to the best model.
    """
    # Get all the models' results:
    models_results = list(path.rglob("*.txt"))
    # txt files are ordered by key, value pairs, f1 score is the first value.

    # Sort the models' results by their f1 score:
    models_results.sort(key=lambda x: float(x.read_text().split(",")[0].split(":")[1].split("\n")[0]), reverse=True)

    # return the name of the best model with the given type:
    return [model for model in models_results if model_type in model.name][0]


def evaluator_main(tree_path: Path = trees_results_path, nn_path: Path = neural_networks_results_path,
                   other_mod_results_path: Path = other_models_results_path, suppress_print=True,
                   balanced: bool = False) -> None:
    """
    Print the results of the best model for each type to the console
    @param tree_path: Path: the path to the trees' results.
    @param nn_path: Path: the path to the neural networks' results.
    @param other_mod_results_path: Path: the path to the other models' results.
    @param suppress_print: bool: whether to suppress the print to the console.
    @param balanced: bool: whether to use the balanced datasets.
    :return: None.
    """
    with open(Path(results_path, 'results_summary.txt'), 'a') as f:
        # get the last folder in the paths:
        tp = tree_path.parts[-1]
        np = nn_path.parts[-1]
        op = other_mod_results_path.parts[-1]
        # write the paths to the summary file:
        f.write(f'Trees path: {tp}\n')
        f.write(f'Neural networks path: {np}\n')
        f.write(f'Other models path: {op}\n')

    if not balanced:
        for alg in ['decision', 'random', 'gradient', 'xgboost', 'neural', 'knn', 'logreg', 'naive', 'svc']:
            if alg in ['decision', 'random', 'gradient', 'xgboost']:
                path = tree_path
            elif alg == "neural":
                path = nn_path
            else:
                path = other_mod_results_path

            best_model = get_best_model_by_type(alg, path)
            best_model_name = best_model.name.split('.')[0].replace('_', ' ').replace("project 2 dataset ", "")\
                .replace("scaling ", "")

            best_score = best_model.read_text().split(",")[0].split(":")[1].split("\n")[0]
            if not suppress_print:
                print(f"The best {best_model_name}, with an f1 score of {round(float(best_score), 3)}")

            # append the results to the results summary file in the results' folder:
            with open(Path(results_path, "results_summary.txt"), "a") as f:
                f.write(f"The best {best_model_name}, with an f1 score of {round(float(best_score), 3)}\n")

        # write a new line to the results summary file:
        with open(Path(results_path, "results_summary.txt"), "a") as f:
            f.write("\n")

    else:  # balanced considers only gradient, convolutional, and support vector machine.
        for alg in ['gradient', 'convolutional', 'svc']:
            if alg == 'gradient':
                path = tree_path
            elif alg == 'convolutional':
                path = nn_path
            elif alg == 'svc':
                path = other_mod_results_path
            else:
                raise ValueError("The algorithm is not supported.")

            best_model = get_best_model_by_type(alg, path)
            best_model_name = best_model.name.split('.')[0].replace('_', ' ').replace("project 2 dataset ", "")\
                .replace("scaling ", "")

            best_score = best_model.read_text().split(",")[0].split(":")[1].split("\n")[0]
            if not suppress_print:
                print(f"The best {best_model_name}, with an f1 score of {round(float(best_score), 3)}")

            # append the results to the results summary file in the results' folder:
            with open(Path(results_path, "results_summary.txt"), "a") as f:
                f.write(f"The best {best_model_name}, with an f1 score of {round(float(best_score), 3)}\n")


if __name__ == '__main__':
    # evaluator_main(trees_results_path, neural_networks_results_path, other_models_results_path, suppress_print=False)
    evaluator_main(trees_balanced_results_path, neural_networks_balanced_results_path,
                   other_models_balanced_results_path, suppress_print=False, balanced=True)