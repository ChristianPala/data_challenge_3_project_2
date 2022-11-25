# neural network for credit card default prediction:
# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import shutil

import pandas as pd
# Modelling:
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    classification_report, confusion_matrix, make_scorer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from modelling.train_test_validation_split import split_data

#  Timing:
from auxiliary.method_timer import measure_time

# Global variables:
results_path: Path = Path("..", "results")
results_path.mkdir(parents=True, exist_ok=True)
neural_network_results_path: Path = Path(results_path, "neural_network")
scaled_datasets_path: Path = Path("..", "data", "scaled_datasets")


# Functions:
def create_model(input_dim: int = 23) -> Sequential:
    """
    This function creates a neural network model.
    :return: Sequential: the model.
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def fit_model(model: Sequential, x_train: np.array, y_train: np.array,  epochs: int = 20) -> Sequential:
    """
    This function fits the model to the training data.
    @param model: Sequential: the model to be fitted.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    @param epochs: int: default = 10: the number of epochs to train the model.
    :return: Fitted model object.
    """
    model.fit(x_train, y_train, epochs=epochs)
    return model


def predict_model(model: Sequential, x_test: np.array) -> np.array:
    """
    This function predicts the target values for the test data.
    @param model: Sequential: the model to be used for prediction.
    @param x_test: np.array: the test data.
    :return: Predicted target values as np.array.
    """
    return model.predict(x_test)


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


def save_evaluation_results(evaluation_results: dict, model_type: str, name_addition: str = None) -> None:
    """
    This function saves the evaluation results to a file.
    @param evaluation_results: dict: the dictionary with the evaluation results.
    @param model_type: str: the type of the model.
    @param name_addition: str: default = None: the name addition to the file name to save the results.
    :return: None. Saves the results to a file in the results' folder.
    """

    # write the results to a file:
    with open(neural_network_results_path / f'{model_type}_base_evaluation_results_{name_addition}.txt', 'w') as f:
        for key, value in evaluation_results.items():
            if key == 'confusion_matrix':
                f.write(f'{key}\n {value}\n')
            elif key == 'classification_report':
                # Todo: Davide, Fabio, should we create a print interface for the classification report?
                # empty line
                f.write('\n')
                value.to_csv(f, mode='a', header=True, sep='\t')
            else:
                f.write(f'{key}: {value}\n')


@measure_time
def neural_network_main() -> None:
    """
    Main method to execute the neural network library.
    :return: None.
    """
    # get all the csv files in the missing_values_handled folder
    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))

    # clean the neural networks results' directory:
    if neural_network_results_path.exists() and neural_network_results_path.is_dir():
        shutil.rmtree(neural_network_results_path, ignore_errors=True)
    neural_network_results_path.mkdir(exist_ok=True, parents=True)
    models_path: Path = Path(neural_network_results_path, "models")
    models_path.mkdir(exist_ok=True, parents=True)

    for csv_file in csv_files:
        # read the csv file:
        df = pd.read_csv(csv_file)

        # split the data into train and test:
        x_train, x_test, y_train, y_test = split_data(df, 'default')

        # if the dataset is not augmented, the size is 23, else it's 26:
        if 'augmented' in csv_file.name:
            input_dim = 26
        else:
            input_dim = 23

        # create the model:
        model = create_model(input_dim=input_dim)

        # fit the model:
        model = fit_model(model, x_train, y_train)

        # predict the target values:
        y_pred = predict_model(model, x_test)

        # evaluate the model:
        evaluation_results = evaluate_model(y_test, y_pred)

        # save the evaluation results:
        save_evaluation_results(evaluation_results=evaluation_results,
                                model_type='neural_network', name_addition=csv_file.stem)

        # save the model:
        model.save(Path(models_path, f'{csv_file.stem}.h5'))


if __name__ == '__main__':
    neural_network_main()

