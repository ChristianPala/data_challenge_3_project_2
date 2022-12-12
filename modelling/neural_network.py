# neural network for credit card default prediction:
# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import shutil
import pandas as pd
import os

# Modelling:
import tensorflow as tf
from keras.utils import plot_model
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Layer, Conv1D, MaxPooling1D, Flatten
from modelling.train_test_validation_split import split_data
from modelling.model_evaluator import save_evaluation_results, evaluate_model
#  Timing:
from auxiliary.method_timer import measure_time
from tqdm import tqdm

# Global variables:
from config import scaled_datasets_path, neural_networks_results_path

if not neural_networks_results_path.exists():
    neural_networks_results_path.mkdir(parents=True, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Functions:
def create_dense_model(input_dim: int = 23) -> Sequential:
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


def create_convolutional_model(input_dim: int = 23) -> Sequential:
    """
    This function creates a convolutional neural network model.
    :return: Convolutional: the model.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
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


def print_model_architecture(model: Sequential) -> None:
    """
    This function prints the model architecture.
    @param model: Sequential: the model to be printed.
    :return: None.
    """
    plot_model(model, to_file=Path(neural_networks_results_path, 'plot.png'), show_shapes=True,
               show_layer_names=True)


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
        model_dense = create_dense_model(input_dim=input_dim)
        model_conv = create_convolutional_model(input_dim=input_dim)

        # fit the model:
        model_dense = fit_model(model_dense, x_train, y_train)
        model_conv = fit_model(model_conv, x_train, y_train)

        # predict the target values:
        y_pred_dense = predict_model(model_dense, x_val)
        y_pred_conv = predict_model(model_conv, x_val)

        # evaluate the model:
        evaluation_results_dense = evaluate_model(y_val, y_pred_dense)
        evaluation_results_conv = evaluate_model(y_val, y_pred_conv)

        # save the evaluation results:
        save_evaluation_results(evaluation_results=evaluation_results_dense,
                                model_type='neural_network_dense', save_path=neural_networks_results_path,
                                dataset_name=csv_file.stem)
        save_evaluation_results(evaluation_results=evaluation_results_conv,
                                model_type='neural_network_convoluted', save_path=neural_networks_results_path,
                                dataset_name=csv_file.stem)

        # save the models:
        model_dense.save(models_path / f'{csv_file.stem}_dense.h5')
        model_conv.save(models_path / f'{csv_file.stem}_convolutional.h5')

        # print the model architecture:
        print_model_architecture(model_dense)
        print_model_architecture(model_conv)


if __name__ == '__main__':
    neural_network_main()



