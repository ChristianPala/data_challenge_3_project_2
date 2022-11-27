# Library to explain the results of the black box neural network model with partial dependence plots:
# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from keras import Model
from keras.models import load_model
from model_explainability.global_surrogate import load_data, import_black_box_model
# Interpretability:
from sklearn.inspection import plot_partial_dependence, PartialDependenceDisplay
# Plotting:
import matplotlib.pyplot as plt
# Global variables:
from config import partial_dependence_results_path


def plot_dependence(feature_name: list[str], black_box_model: Model, x_train: np.array, y_train: np.array) -> None:
    """
    This function plots the partial dependence of a feature.
    @param feature_name: str: the name of the feature.
    @param black_box_model: keras.Model: the black box model.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: None
    """
    plot_partial_dependence(black_box_model, x_train, feature_name, target=y_train, n_jobs=-1)
    plt.savefig(Path(partial_dependence_results_path, f"{feature_name}.png"))
    plt.close()


def pdp_main() -> None:
    """
    This function is the main function.
    :return: None
    """
    x_train, y_train, x_test, y_test = load_data()
    black_box_model = import_black_box_model()

    # plot the partial dependence of all features:
    for feature_name in x_train.columns:
        plot_dependence(feature_name, black_box_model, x_train, y_train)

    # examine limit_bal and pay_status_total combined:
    plot_dependence(['limit_bal', 'pay_status_total'], black_box_model, x_train, y_train)

    """
    Not tested yet:
    """




