
# Libraries:
import pandas as pd
import numpy as np

# Local Interpretable Model-agnostic Explanations framework
# import lime
from lime.lime_tabular import LimeTabularExplainer


# Data manipulation:
from pathlib import Path

# Splitting the data:
from modelling.train_test_validation_split import split_data


# Global variables:
data_path = Path("..", "data")


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
                                     feature_names=x_train.columns.values.tolist(),
                                     class_names=[target],
                                     mode='classification',
                                     random_state=random_state)

    # Choose the j_th instance and use it to predict the results
    exp = explainer.explain_instance(
        data_row=x_test.iloc[j],
        predict_fn=model.predict_proba
    )

    # Show the predictions
    exp.show_in_notebook(show_table=True)

    # Code for SP-LIME
    # import warnings
    # from lime import submodular_pick
    #
    # # Remember to convert the dataframe to matrix values
    # # SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model
    # sp_obj = submodular_pick.SubmodularPick(explainer, df_titanic[model.feature_name()].values, \
    #                                         prob, num_features=5, num_exps_desired=10)
    #
    # [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]

    # Visualize features importance for wrong predictors
    # preds = lr.predict(X_test)
    #
    # false_preds = np.argwhere((preds != Y_test)).flatten()
    #
    # idx = random.choice(false_preds)
    #
    # print("Prediction : ", breast_cancer.target_names[lr.predict(X_test[idx].reshape(1, -1))[0]])
    # print("Actual :     ", breast_cancer.target_names[Y_test[idx]])
    #
    # explanation = explainer.explain_instance(X_test[idx], lr.predict_proba)
    #
    # explanation.show_in_notebook()


