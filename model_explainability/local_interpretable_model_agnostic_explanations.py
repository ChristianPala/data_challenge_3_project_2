# Interpret our model locally with LIME:
# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
import random
from pathlib import Path
# Plotting:
import matplotlib.pyplot as plt
# Explaining:
# Local Interpretable Model-agnostic Explanations framework
# using the library from https://lime-ml.readthedocs.io/en/latest/
from lime.lime_tabular import LimeTabularExplainer
# Splitting the data:
from modelling.train_test_validation_split import split_data

# Global variables:
from config import data_path


# Functions:
def lime(df: pd.DataFrame, target: str, model: ..., j: int = 5, random_state: int = 42) -> None:
    # TODO: not sure which type of data the function should expect from the model parameter
    """
    This function carry out a Local Model-agnostic Explanation of a pre-trained model.
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param j: int: index of the data our model will use for the prediction.
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    """

    # Splitting the data:
    x_train, x_test, y_train, y_test = split_data(df, target)

    # LIME explainer
    explainer = LimeTabularExplainer(training_data=x_train.values,
                                     feature_names=x_train.columns,
                                     class_names=['did not default', 'default'],
                                     mode='classification',
                                     random_state=random_state)

    # Choose the j_th instance and use it to predict the results
    exp = explainer.explain_instance(
        data_row=x_test.iloc[j],
        predict_fn=model.predict_proba
    )

    # Save the predictions
    exp.save_to_file(Path(data_path, 'lime_report.html'))

    # Analyze wrong predictions
    y_pred = model.predict(x_test)

    wrong_pred = np.argwhere((y_pred != y_test.to_numpy())).flatten()

    idx = random.choice(wrong_pred)

    # print("Prediction : ", model.predict(x_test.to_numpy()[idx].reshape(1, -1))[0])
    # print("Actual :     ", y_test.iloc[idx])

    explanation = explainer.explain_instance(x_test.iloc[idx], model.predict_proba)

    explanation.save_to_file(Path(data_path, 'lime_report_wrong_pred.html'))
