# Auxiliary library for splitting the data into train, test and validation sets using
# sklearn.model_selection.train_test_split.
from typing import Union, Tuple, Any

# Libraries:
import pandas as pd
import numpy as np


# Data manipulation:
from pathlib import Path

# Splitting the data:
from sklearn.model_selection import train_test_split


# Global variables:
data_path = Path("..", "data")


# Functions:
def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42,
               validation=False) \
        -> Union[tuple[np.array, np.array, np.array, np.array], tuple[np.array, np.array, np.array, np.array, np.array,
                                                                      np.array]]:
    """
    This function splits the data into train, validation and testing.
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param val_size: float: default = 0.2: the percentage of the data to be used for validation from the training set.
    @param test_size: float: default = 0.2: the percentage of the data to be used for testing from the whole dataset.
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    @param validation: bool: default = False: whether to split the data into train and validation sets.
    :return: x_train, x_val, x_test, y_train, y_val, y_test as np.arrays.
    """
    # Splitting the data:
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                        df[target],
                                                        test_size=test_size,
                                                        random_state=random_state)
    if not validation:
        return X_train, X_test, y_train, y_test

    # Splitting the test set into test and validation sets:
    X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                    y_test,
                                                    test_size=val_size,
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

