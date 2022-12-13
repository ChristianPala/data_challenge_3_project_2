# File to estimate which balancing procedure is the best for the dataset

# Data manipulation
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

from config import trees_balanced_results_path, neural_networks_balanced_results_path,\
    other_models_balanced_results_path


def get_f_1_score(text_path: Path) -> float:
    """
    This function gets the f1 score from the text file.
    @param text_path: Path: the path to the text file.
    :return: float: the f1 score.
    """
    with open(text_path, 'r') as f:
        for line in f:
            if line.startswith('f1'):
                return float(line.split(': ')[1])


def get_all_models_results(path: Path) -> list:
    """
    This function gets all the models' results.
    @param path: Path: the path to the results' subfolder.
    :return: list: a list of all the models' results.
    """
    # Get all the models' results:
    models_results = list(path.rglob("*.txt"))
    return models_results


def compare_methods(path: Path) -> Path:
    """
    This function compares the methods and saves the best method.
    @param path: Path: the path to the results' subfolder.
    :return: None.
    """
    # get all the models' results:
    models_results = get_all_models_results(path)
    # sort the models' results by their f1 score:
    models_results.sort(key=lambda x: get_f_1_score(x))
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
    tree_f1_score: float = get_f_1_score(tree)
    neural_network_f1_score: float = get_f_1_score(neural_network)
    other_model_f1_score: float = get_f_1_score(other_model)
    # get the best model:
    best_model: Path = tree if tree_f1_score > neural_network_f1_score and tree_f1_score > other_model_f1_score else \
        neural_network if neural_network_f1_score > other_model_f1_score else other_model

    return best_model


def main() -> None:
    """
    This function compares the methods and saves the best method.
    :return: None.
    """
    # compare the methods:
    tree: Path = compare_methods(trees_balanced_results_path)
    print(f"Best tree model: {tree}")
    neural_network: Path = compare_methods(neural_networks_balanced_results_path)
    print(f"Best neural network model: {neural_network}")
    other_model: Path = compare_methods(other_models_balanced_results_path)
    print(f"Best other model: {other_model}")
    # get the best overall model:
    best_model: Path = get_best_overall_model(tree, neural_network, other_model)
    print(f"Best overall model: {best_model}")
    # store the best model in the best_models sub-folder:
    best_models_path: Path = Path(best_model.parent, "best_models")
    best_models_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model, best_models_path)


if __name__ == '__main__':
    main()
