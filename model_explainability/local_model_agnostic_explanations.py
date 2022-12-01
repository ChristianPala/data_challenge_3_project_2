# Libraries:
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from config import *

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


# def return_weights(exp):
#     """
#     Get weights from LIME explanation object
#     Based on this implementation:
#     https://github.com/conorosully/medium-articles/blob/master/src/interpretable%20ml/LIME/lime_tutorial.ipynb
#     """
#
#     exp_list = exp.as_map()[1]
#     exp_list = sorted(exp_list, key=lambda x: x[0])
#     exp_weight = [x[1] for x in exp_list]
#
#     return exp_weight


# Functions:
def lime_explanation(training: pd.DataFrame, testing: pd.DataFrame, target: str, model: ..., j: int = 5,
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
    # x_train, x_test, y_train, y_test = split_data(df, target)
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    x_test = testing.drop("default", axis=1)
    y_test = testing["default"]

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

    # Save the report
    # exp.save_to_file('../model_explainability/results/lime_report.html')
    exp.save_to_file(Path(lime_local_explanation_results_path, 'lime_report.html'))

    if with_wrong_prediction_analysis:
        # Analyze wrong predictions
        y_pred = model.predict(x_test)

        # wrong prediction indexes
        # wrong_pred = np.argwhere((y_pred != y_test.to_numpy())).flatten()

        tp_idxs = np.where((y_test == 1) & (y_pred == 1))[0]
        tn_idxs = np.where((y_test == 0) & (y_pred == 0))[0]
        fp_idxs = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_idxs = np.where((y_test == 1) & (y_pred == 0))[0]

        # choose a random index
        tp_idx = random.choice(tp_idxs)
        tn_idx = random.choice(tn_idxs)
        fp_idx = random.choice(fp_idxs)
        fn_idx = random.choice(fn_idxs)

        # print("Prediction : ", model.predict(x_test.to_numpy()[idx].reshape(1, -1))[0])
        # print("Actual :     ", y_test.iloc[idx])

        # explain the wrongly predicted instance
        explanation = explainer.explain_instance(x_test.iloc[tp_idx], model.predict_proba)

        # save the html file
        # explanation.save_to_file(Path(project_root_path, 'model_explainability',  'results',
        #                               'lime_report_tp_pred.html'))
        explanation.save_to_file(Path(lime_local_explanation_results_path, 'lime_report_tp_pred.html'))

        # explain the wrongly predicted instance
        explanation = explainer.explain_instance(x_test.iloc[tn_idx], model.predict_proba)

        # save the html file
        # explanation.save_to_file(Path(project_root_path, 'model_explainability', 'results',
        #                               'lime_report_tn_pred.html'))
        explanation.save_to_file(Path(lime_local_explanation_results_path, 'lime_report_tn_pred.html'))

        # explain the wrongly predicted instance
        explanation = explainer.explain_instance(x_test.iloc[fp_idx], model.predict_proba)

        # save the html file
        # explanation.save_to_file(Path(project_root_path, 'model_explainability', 'results',
        #                               'lime_report_fp_pred.html'))
        explanation.save_to_file(Path(lime_local_explanation_results_path, 'lime_report_fp_pred.html'))

        # explain the wrongly predicted instance
        explanation = explainer.explain_instance(x_test.iloc[fn_idx], model.predict_proba)

        # save the html file
        # explanation.save_to_file(Path(project_root_path, 'model_explainability', 'results',
        #                               'lime_report_fn_pred.html'))
        explanation.save_to_file(Path(lime_local_explanation_results_path, 'lime_report_fn_pred.html'))


def shap_explanation(training: pd.DataFrame, testing: pd.DataFrame, target: str, model: ..., j: int = 5,
                     with_global_summary_plots: bool = False, random_state: int = 42) -> None:
    """
    This function carry out a Local Model-agnostic Explanation of a pre-trained model using the shap library:
    https://shap.readthedocs.io/en/latest/index.html
    @param df: pd.DataFrame: the dataset to be split.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param j: int: index of the data our model will use for the prediction.
    @param with_global_summary_plots: weather or not to create the analysis of a wrongly predicted instance
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    """

    # Splitting the data:
    # x_train, x_test, y_train, y_test = split_data(df, target)
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    x_test = testing.drop("default", axis=1)
    y_test = testing["default"]

    # y_pred = model.predict(x_test)

    # initialize the shapley values:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    shap.waterfall_plot(shap.Explanation(values=shap_values[0][j],
                                         base_values=explainer.expected_value[0], data=x_test.iloc[j],
                                         feature_names=x_test.columns.tolist())
                        , show=False)

    # give more space for the y-axis:
    plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.2)
    plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.2)
    # increase the size of the plot:
    plt.gcf().set_size_inches(10, 5)
    # plt.savefig(Path(project_root_path, 'model_explainability', 'results', 'shap_waterfall.png'))
    plt.savefig(Path(shap_local_explanation_results_path, 'shap_waterfall.png'))
    plt.close()

    shap.force_plot(explainer.expected_value[0], shap_values[0][j], x_test.values[j], feature_names=x_test.columns,
                    matplotlib=True, show=False)

    # save the plot:
    # plt.savefig(Path(project_root_path, 'model_explainability', 'results', 'shap_force_plot.png'),
    #             dpi=300, bbox_inches='tight')
    plt.savefig(Path(shap_local_explanation_results_path, 'shap_force_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    if with_global_summary_plots:
        # shap.summary_plot(shap_values, x_test.values, feature_names=x_test.columns, show=True)
        shap.summary_plot(shap_values, x_test.values, plot_type="bar", class_names=['did not default', 'default'],
                          feature_names=x_test.columns, show=False)

        # save the plot:
        # plt.savefig(Path(project_root_path, 'model_explainability', 'results', 'shap_summary_plot.png'))
        plt.savefig(Path(shap_local_explanation_results_path, 'shap_summary_plot.png'))
        plt.close()

        shap.summary_plot(shap_values[0], x_test.values, feature_names=x_test.columns, show=False)

        # save the plot:
        # plt.savefig(Path(project_root_path, 'model_explainability', 'results',
        #                  'shap_summary_plot_class_not_default.png'))
        plt.savefig(Path(shap_local_explanation_results_path, 'shap_summary_plot_class_not_default.png'))
        plt.close()

        shap.summary_plot(shap_values[1], x_test.values, feature_names=x_test.columns, show=False)

        # save the plot:
        # plt.savefig(Path(project_root_path, 'model_explainability', 'results', 'shap_summary_plot_class_default.png'))
        plt.savefig(Path(shap_local_explanation_results_path, 'shap_summary_plot_class_default.png'))
        plt.close()
