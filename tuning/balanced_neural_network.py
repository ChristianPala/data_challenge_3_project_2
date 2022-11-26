# Library to test the results of the neural network with balanced data.
import pandas as pd

from modelling.neural_network import evaluate_model, predict_model, fit_model, create_model

# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import shutil


# Functions:
def load_train_validation_testing(folder: Path) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    This function loads the training, validation and testing data from the given folder.
    @param folder: Path: the folder containing the data.
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame: the training, validation and testing data.
    """
    # get all the csv files in the scaled_datasets folder:
    csv_files: list[Path] = list(folder.glob('*.csv'))
    # create a list to store the dataframes:
    data: list[tuple[Path, Path, Path]] = []

    # loop through the csv files:
    for csv_file in csv_files:
        # get the name of the csv file:
        csv_file_name: str = csv_file.stem
        #



    # create a list of tuples of the dataframes:
    dataframes: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    # iterate over the tuples of paths:
    for training, validation, testing in data:
        # load the dataframes:
        training_df: pd.DataFrame = pd.read_csv(training)
        validation_df: pd.DataFrame = pd.read_csv(validation)
        testing_df: pd.DataFrame = pd.read_csv(testing)
        # append the dataframes to the list:
        dataframes.append((training_df, validation_df, testing_df))
    # return the dataframes:
    return dataframes