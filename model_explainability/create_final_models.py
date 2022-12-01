# Auxiliary library to create and train the final models from the tuning results:
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
# Modelling:
from sklearn.ensemble import GradientBoostingClassifier
from modelling.neural_network import fit_model, create_model
from sklearn.svm import SVC

# Global variables:
from config import final_models_path, final_training_csv_path
final_models_path.mkdir(parents=True, exist_ok=True)


def gradient_boosting_model(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    This function creates a gradient boosting model.
    :return: None
    """
    gradient_booster = GradientBoostingClassifier(random_state=42)  # Todo add all parameters from the tuning results
    gradient_booster.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(gradient_booster, Path(final_models_path, "gradient_boosting_model.pkl"))


def neural_network_model(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    This function creates a neural network model.
    :return: None
    """
    # create the model:
    model = create_model(x_train.shape[1])  # Todo add all parameters from the tuning results
    # fit the model:
    model = fit_model(model, x_train, y_train)
    # save the model:
    model.save(Path(final_models_path, "neural_network_model.h5"))
    # save the model architecture:
    with open(Path(final_models_path, "neural_network_model_architecture.json"), "w") as json_file:
        json_file.write(model.to_json())
    # save the model weights:
    model.save_weights(Path(final_models_path, "neural_network_model_weights.h5"))


def supper_vector_machine_model(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    This function creates a support vector machine model.
    :return: None
    """
    svc = SVC(random_state=42)  # Todo add all parameters from the tuning results
    svc.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(svc, Path(final_models_path, "supper_vector_machine_model.pkl"))


def main() -> None:
    """
    This function is the main function.
    :return: None
    """
    training = pd.read_csv(final_training_csv_path)
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    gradient_boosting_model(x_train, y_train)
    neural_network_model(x_train, y_train)
    supper_vector_machine_model(x_train, y_train)


if __name__ == '__main__':
    main()

