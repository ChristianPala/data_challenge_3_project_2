# Auxiliary library to create and train the final models from the tuning results:
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
from keras import Model
# Modelling:
from sklearn.ensemble import GradientBoostingClassifier

from modelling.neural_network import fit_model, create_convolutional_model
from sklearn.svm import SVC

# Global variables:
from config import final_train_csv_path, final_val_csv_path, final_models_path

final_models_path.mkdir(parents=True, exist_ok=True)


def gradient_boosting_model(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    This function creates a gradient boosting model.
    :return: None
    """
    gb = GradientBoostingClassifier(learning_rate=0.22271924404121154, loss='exponential', max_depth=7,
                                    max_leaf_nodes=4, min_impurity_decrease=0.35176743416487977,
                                    min_samples_leaf=1, min_samples_split=5,
                                    min_weight_fraction_leaf=0.000723912587745684, n_estimators=593,
                                    subsample=0.7424194687719635, random_state=42)

    gb.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(gb, Path(final_models_path, "gradient_boosting_model.pkl"))


def neural_network_model(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    This function creates a neural network model.
    :return: None
    """
    # create the model:
    model = create_convolutional_model(x_train.shape[1])  # Todo add all parameters from the tuning results
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
    svc = SVC(random_state=42, C=4900, gamma=10 ** -5)
    svc.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(svc, Path(final_models_path, "supper_vector_machine_model.pkl"))


def create_final_models_main() -> None:
    """
    This function is the main function.
    :return: None
    """
    # load the data:
    training = pd.read_csv(final_train_csv_path)
    validation = pd.read_csv(final_val_csv_path)

    # concatenate training and validation:
    training = pd.concat([training, validation], axis=0)

    # create the final models:
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    # create and save the models:
    gradient_boosting_model(x_train, y_train)
    neural_network_model(x_train, y_train)
    supper_vector_machine_model(x_train, y_train)


if __name__ == '__main__':
    create_final_models_main()
