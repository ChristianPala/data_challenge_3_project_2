# Libraries:
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import shap
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots


from config import *


# Data manipulation:
from pathlib import Path

# Splitting the data:
from modelling.train_test_validation_split import split_data


# Global variables:
data_path = Path("..", "data")


# Functions:
def permutation_feature_importance(training: pd.DataFrame, testing: pd.DataFrame, target: str, model: ..., random_state: int = 42) -> None:
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
    # x_train, x_test, y_train, y_test = split_data(df, target)
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    x_test = testing.drop("default", axis=1)
    y_test = testing["default"]

    perm = PermutationImportance(model, random_state=random_state, scoring='f1').fit(x_test, y_test)
    html_obj = eli5.show_weights(perm, feature_names=x_test.columns.tolist())

    # Write html object to a file (adjust file path; Windows path is used here)
    with open(Path(permutation_fi_results_path, 'permutation_feature_importance.html'), 'wb') as f:
        f.write(html_obj.data.encode("UTF-8"))

    # # Initialize a list of results
    # results = []
    # # Iterate through each predictor
    # for predictor in X_test:
    #     # Create a copy of X_test
    #     X_test_copy = X_test.copy()
    #
    #     # Scramble the values of the given predictor
    #     X_test_copy[predictor] = X_test[predictor].sample(frac=1).values
    #
    #     # Calculate the new RMSE
    #     new_rmse = mean_squared_error(regr.predict(X_test_copy), y_test,
    #                                   squared=False)
    #
    #     # Append the increase in MSE to the list of results
    #     results.append({'pred': predictor,
    #                     'score': new_rmse - rmse_full_mod})
    # # Convert to a pandas dataframe and rank the predictors by score
    # resultsdf = pd.DataFrame(results).sort_values(by='score',
    #                                               ascending=False)


def partial_dependence_plots(training: pd.DataFrame, testing: pd.DataFrame, target: str, model: ...,
                             predict_proba_available: bool = True,
                             random_state: int = 42) -> None:
    """
    This function carry out a Global Model-agnostic Explanation of a pre-trained model.
    We displays (global) functional relationship between a set of features and the target variable.
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    """

    def predict_use_keras(x):
        prob = model.predict(x)
        prob = pd.Series(map(lambda x: x[0], prob))

        n_prob = prob.apply(lambda x: 1 - x)
        probs = np.column_stack((n_prob, prob))
        return probs

    # Splitting the data:
    # x_train, x_test, y_train, y_test = split_data(df, target)
    x_train = training.drop(target, axis=1)
    y_train = training[target]

    x_test = testing.drop(target, axis=1)
    y_test = testing[target]

    # Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
    # features_to_plot = ['‘Goal Scored’', '‘Distance Covered(Kms)’']
    # inter1 = pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)
    # pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type=’contour’)
    # plt.show()

    if predict_proba_available:
        for name in x_test.columns:
            shap.plots.partial_dependence(name, model.predict, x_test, ice=False, model_expected_value=True,
                                          feature_expected_value=True, show=False)
            # save the plot:
            plt.savefig(Path(partial_dependence_results_path, f'shap_pdp_{name}.png'))
            plt.close()
    else:
        for name in x_test.columns:
            shap.plots.partial_dependence(name, predict_use_keras, x_test, ice=False, model_expected_value=True,
                                          feature_expected_value=True, show=False)
            # save the plot:
            plt.savefig(Path(partial_dependence_results_path, f'shap_pdp_{name}.png'))
            plt.close()


