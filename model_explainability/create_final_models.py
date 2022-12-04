# Auxiliary library to create and train the final models from the tuning results:
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
from keras import Model
# Modelling:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from modelling.neural_network import fit_model, create_convolutional_model
from sklearn.svm import SVC

# Global variables:
from config import final_training_oversampled_csv_path, final_validation_oversampled_csv_path, \
    final_training_undersampled_csv_path, final_validation_undersampled_csv_path, \
    final_train_tomek_csv_path, final_val_tomek_csv_path, \
    final_models_path
final_models_path.mkdir(parents=True, exist_ok=True)


def gradient_boosting_model(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    This function creates a gradient boosting model.
    :return: None
    """
    gradient_booster = GradientBoostingClassifier(random_state=42, n_estimators=500,
                                                  max_depth=4, learning_rate=0.1, min_samples_leaf=1,
                                                  min_samples_split=6)

    gradient_booster.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(gradient_booster, Path(final_models_path, "gradient_boosting_model.pkl"))


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
    training_o = pd.read_csv(final_training_oversampled_csv_path)
    validation_o = pd.read_csv(final_validation_oversampled_csv_path)
    training_u = pd.read_csv(final_training_undersampled_csv_path)
    validation_u = pd.read_csv(final_validation_undersampled_csv_path)
    training_t = pd.read_csv(final_train_tomek_csv_path)
    validation_t = pd.read_csv(final_val_tomek_csv_path)
    # merge the training and validation data:
    training_o = pd.concat([training_o, validation_o], ignore_index=True)
    training_u = pd.concat([training_u, validation_u], ignore_index=True)
    training_t = pd.concat([training_t, validation_t], ignore_index=True)
    # split the data:
    x_train_o = training_o.drop("default", axis=1)
    y_train_o = training_o["default"]
    x_train_u = training_u.drop("default", axis=1)
    y_train_u = training_u["default"]
    x_train_t = training_t.drop("default", axis=1)
    y_train_t = training_t["default"]
    # create and save the models:
    gradient_boosting_model(x_train_t, y_train_t)
    neural_network_model(x_train_u, y_train_u)
    supper_vector_machine_model(x_train_o, y_train_o)


if __name__ == '__main__':
    create_final_models_main()

