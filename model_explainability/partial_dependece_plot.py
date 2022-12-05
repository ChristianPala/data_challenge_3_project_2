# Library to explain the results of the black box neural network model with partial dependence plots:
# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from keras.models import Model, load_model
# Interpretability:
from sklearn.inspection import PartialDependenceDisplay
# Plotting:
import matplotlib.pyplot as plt
# Global variables:
from config import final_train_csv_path, final_val_csv_path, partial_dependence_results_path, final_neural_network_path, \
    final_models_path

partial_dependence_results_path.mkdir(parents=True, exist_ok=True)


def plot_dependence(feature_name: list[str], black_box_model: Model, x_train: np.array) -> None:
    """
    This function plots the partial dependence of a feature.
    @param feature_name: str: the name of the feature.
    @param black_box_model: keras.Model: the black box model.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: None
    """
    # create the partial dependence plot:
    pdp = PartialDependenceDisplay.from_estimator(black_box_model, x_train, [feature_name])
    # plot the partial dependence plot:
    pdp.plot()
    # save the plot:
    plt.savefig(Path(partial_dependence_results_path, f"{feature_name}.png"))
    # close the plot:
    plt.close()


def pdp_main() -> None:
    """
    This function is the main function.
    :return: None
    """
    training = pd.read_csv(final_train_csv_path)
    validation = pd.read_csv(final_val_csv_path)

    # concatenate the training and validation data:
    training = pd.concat([training, validation], axis=0)
    x_train = training.drop(columns=["default"])

    # import the gradient boosting model:
    gradient_boosting_model = pd.read_pickle(Path(final_models_path, "gradient_boosting_model.pkl"))

    # plot the partial dependence of all features:
    for feature_name in x_train.columns:
        plot_dependence(feature_name, gradient_boosting_model, x_train)
        plt.close()

    # examine limit_bal and pay_status_total combined:
    plot_dependence(['limit_bal', 'pay_status_total'], gradient_boosting_model, x_train)
    plt.close()


if __name__ == '__main__':
    pdp_main()
