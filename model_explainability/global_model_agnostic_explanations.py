# Libraries:
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance


from config import *


# Data manipulation:
from pathlib import Path

# Splitting the data:
from modelling.train_test_validation_split import split_data


# Global variables:
data_path = Path("..", "data")


# Functions:
def permutation_feature_importance(df: pd.DataFrame, target: str, model: ..., random_state: int = 42) -> None:
    """
    This function carry out a Global Model-agnostic Explanation of a pre-trained model using the eli5 framework:
    https://eli5.readthedocs.io/en/latest/overview.html
    We will shuffle the values in a single column, make predictions using the resulting dataset.
    Use these predictions and the true target values to calculate how much the loss function suffered from shuffling.
    That performance deterioration measures the importance of the variable you just shuffled.
    We will go back to the original data order (undoing the previous shuffle) and repeat the procedure with the next
    column in the dataset, until we have calculated the importance of each column.
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    """

    # Splitting the data:
    x_train, x_test, y_train, y_test = split_data(df, target)

    perm = PermutationImportance(model, random_state=random_state).fit(x_test, y_test)
    html_obj = eli5.show_weights(perm, feature_names=x_test.columns.tolist())

    # Write html object to a file (adjust file path; Windows path is used here)
    with open(Path(project_root_path, 'model_explainability', 'results', 'permutation_feature_importance.html'),
              'wb') as f:
        f.write(html_obj.data.encode("UTF-8"))


