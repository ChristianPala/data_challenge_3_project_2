# Libraries:
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Local Interpretable Model-agnostic Explanations frameworks imports
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap


# Data manipulation:
from pathlib import Path

# Splitting the data:
from modelling.train_test_validation_split import split_data


# Global variables:
data_path = Path("..", "data")


def return_weights(exp):
    """
    Get weights from LIME explanation object
    Based on this implementation:
    https://github.com/conorosully/medium-articles/blob/master/src/interpretable%20ml/LIME/lime_tutorial.ipynb
    """

    exp_list = exp.as_map()[1]
    exp_list = sorted(exp_list, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_list]

    return exp_weight


# Functions:
def lime__explanation(df: pd.DataFrame, target: str, model: ..., j: int = 5,
                      with_wrong_prediction_analysis: bool = False, random_state: int = 42) -> None:
    # TODO: not sure which type of data the function should expect from the parameter model
    """
    This function carry out a Local Model-agnostic Explanation of a pre-trained model using the lime framework:
    https://github.com/marcotcr/lime/blob/master/citation.bib
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param j: int: index of the data our model will use for the prediction.
    @param with_wrong_prediction_analysis: weather or not to create the analysis of a wrongly predicted instance
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
    exp.save_to_file('../model_explainability/results/lime_report.html')

    # # Compute the weights
    # weights = []
    #
    # # Iterate over first 100 rows in feature matrix
    # for x in x_test.values[0:15]:
    #     # Get explanation
    #     # Get explanation
    #     exp = explainer.explain_instance(x,
    #                                      model.predict,
    #                                      num_features=10,
    #                                      labels=x_test.columns)
    #
    #     # Get weights
    #     exp_weight = return_weights(exp)
    #     weights.append(exp_weight)
    #
    # # Create DataFrame
    # lime_weights = pd.DataFrame(data=weights, columns=x_test.columns)
    #
    # # Get abs mean of LIME weights
    # abs_mean = lime_weights.abs().mean(axis=0)
    # abs_mean = pd.DataFrame(data={'feature': abs_mean.index, 'abs_mean': abs_mean})
    # abs_mean = abs_mean.sort_values('abs_mean')
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    #
    # y_ticks = range(len(abs_mean))
    # y_labels = abs_mean.feature
    #
    # # plot scatterplot for each feature
    # for i, feature in enumerate(y_labels):
    #     feature_weigth = lime_weights[feature]
    #     feature_value = x_test[feature][0:100]
    #
    #     plt.scatter(x=feature_weigth,
    #                 y=[i] * len(feature_weigth),
    #                 c=feature_value,
    #                 cmap='bwr',
    #                 edgecolors='black',
    #                 alpha=0.8)
    #
    # plt.vlines(x=0, ymin=0, ymax=9, colors='black', linestyles="--")
    # plt.colorbar(label='Feature Value', ticks=[])
    #
    # plt.yticks(ticks=y_ticks, labels=y_labels, size=15)
    # plt.xlabel('LIME Weight', size=20)
    #
    # plt.savefig('results/lime_weights_beeswarm.png', dpi=200, bbox_inches='tight')

    if with_wrong_prediction_analysis:
        # Analyze wrong predictions
        y_pred = model.predict(x_test)

        # wrong prediction indexes
        wrong_pred = np.argwhere((y_pred != y_test.to_numpy())).flatten()

        # choose a random index
        idx = random.choice(wrong_pred)

        # print("Prediction : ", model.predict(x_test.to_numpy()[idx].reshape(1, -1))[0])
        # print("Actual :     ", y_test.iloc[idx])

        # explain the wrongly predicted instance
        explanation = explainer.explain_instance(x_test.iloc[idx], model.predict_proba)

        # save the html file
        explanation.save_to_file('results/lime_report_wrong_pred.html')




def shap_explanation(df: pd.DataFrame, target: str, model: ..., j: int = 5,
                     with_wrong_prediction_analysis: bool = False, random_state: int = 42) -> None:
    """
    This function carry out a Local Model-agnostic Explanation of a pre-trained model using the shap library:
    https://shap.readthedocs.io/en/latest/index.html
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param j: int: index of the data our model will use for the prediction.
    @param with_wrong_prediction_analysis: weather or not to create the analysis of a wrongly predicted instance
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    """

    # Splitting the data:
    x_train, x_test, y_train, y_test = split_data(df, target)
