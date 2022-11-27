# neural network for credit card default prediction:
# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import shutil
import pandas as pd

# Modelling:
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Layer
from modelling.train_test_validation_split import split_data
from modelling.model_evaluator import save_evaluation_results, evaluate_model
#  Timing:
from auxiliary.method_timer import measure_time
from tqdm import tqdm

# Global variables:
from config import scaled_datasets_path, neural_networks_results_path

if not neural_networks_results_path.exists():
    neural_networks_results_path.mkdir(parents=True, exist_ok=True)


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

# TODO to debug
def create_model_with_layers(model: Model, layers: list[Layer], dropout: float = 0.1, optimizer: str = "adam", loss: str = 'binary_crossentropy', metrics: list[str] = ['accuracy']) -> Model:
    compiled_model = model
    for i in range(len(layers)):
        compiled_model.add(layers[i])
        if i < len(layers)-1:
            compiled_model.add(Dropout(dropout))

    compiled_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return compiled_model


def fit_model(model: Sequential, x_train: np.array, y_train: np.array,  epochs: int = 20) -> Sequential:
    """
    This function fits the model to the training data.
    @param model: Sequential: the model to be fitted.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    @param epochs: int: default = 10: the number of epochs to train the model.
    :return: Fitted model object.
    """
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model


def predict_model(model: Sequential, x_test: np.array) -> np.array:
    """
    This function predicts the target values for the test data.
    @param model: Sequential: the model to be used for prediction.
    @param x_test: np.array: the test data.
    :return: Predicted target values as np.array.
    """
    return model.predict(x_test, verbose=0)


@measure_time
def neural_network_main() -> None:
    """
    Main method to execute the neural network library.
    :return: None.
    """
    if not scaled_datasets_path.exists():
        raise FileNotFoundError('The scaled datasets folder and files do not exist. Create them first.')

    # clean the neural_networks_results folder:
    if neural_networks_results_path.exists() and neural_networks_results_path.is_dir():
        shutil.rmtree(neural_networks_results_path)
    neural_networks_results_path.mkdir(parents=True, exist_ok=True)
    models_path = neural_networks_results_path / 'models'

    # get all the csv files in the scaled_datasets folder:
    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))

    for csv_file in tqdm(csv_files, desc='Neural networks', unit='file', total=len(csv_files), colour='green'):
        # read the csv file:
        df = pd.read_csv(csv_file)

        # split the data into train and test:
        x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)

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
        y_pred = predict_model(model, x_val)

        # evaluate the model:
        evaluation_results = evaluate_model(y_val, y_pred)

        # save the evaluation results:
        save_evaluation_results(evaluation_results=evaluation_results,
                                model_type='neural_network', save_path=neural_networks_results_path,
                                dataset_name=csv_file.stem)

        # save the model:
        model.save(models_path / f'{csv_file.stem}.h5')

        # save the model:
        model.save(Path(models_path, f'{csv_file.stem}.h5'))


if __name__ == '__main__':
    neural_network_main()


