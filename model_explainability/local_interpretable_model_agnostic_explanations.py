# Interpret our model locally with LIME: https://github.com/marcotcr/lime
# and SHAP https: https://shap.readthedocs.io/en/latest/
# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
from pathlib import Path
import os
# Randomness:
import random
# Plotting:
import matplotlib.pyplot as plt
# Local Interpretable Model-agnostic Explanations frameworks imports
from lime.lime_tabular import LimeTabularExplainer
# SHAPELY:
import shap
from shap import maskers
# Type hinting:
from typing import Literal
# Timing:
from auxiliary.method_timer import measure_time
# Keras model loading:
from keras.models import load_model
# Deprecation warnings:
import warnings

# Global variables:
from config import final_models_path, final_test_under_csv_path, final_train_under_csv_path, final_val_under_csv_path, \
    shap_results_path, lime_results_path, global_surrogate_models_path, global_surrogate_results_path
# Ensure the folders exist:
shap_results_path.mkdir(parents=True, exist_ok=True)
lime_results_path.mkdir(parents=True, exist_ok=True)

# Tensorflow and scikit learn logging:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


# Functions:
def lime_explanation(training: pd.DataFrame, testing: pd.DataFrame, target: str, model: ...,
                     model_name: str, j: int = 0,
                     with_wrong_prediction_analysis: bool = False, random_state: int = 42) -> None:
    """
    This function carry out a Local Model-agnostic Explanation of a pre-trained model using the lime framework:
    https://github.com/marcotcr/lime/blob/master/citation.bib
    @param training: pd.DataFrame: the training dataset.
    @param testing: pd.DataFrame: the testing dataset.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param model_name: str: the name of out model, to save appropriately the results.
    @param j: int: index of the data our model will use for the prediction.
    @param with_wrong_prediction_analysis: weather or not to create the report of a random wrongly predicted instance.
    @param random_state: int: default = 42: the random state to be used for reproducibility.
    """

    # Splitting the data:
    x_train = training.drop(target, axis=1)

    x_test = testing.drop(target, axis=1)
    y_test = testing[target]

    # Check if the function for the return of the classes probabilities is available
    if hasattr(model, 'predict_proba'):
        predict_proba_available = True
    else:
        predict_proba_available = False

    # Function for sequential keras model that with the predict method returns only the prob. of the positive class
    def predict_use_keras(x):
        prob = model.predict(x)
        prob = pd.Series(map(lambda y: y[0], prob))

        n_prob = prob.apply(lambda y: 1 - y)
        probs = np.column_stack((n_prob, prob))
        return probs

    # LIME explainer
    explainer = LimeTabularExplainer(training_data=x_train.values,
                                     feature_names=x_train.columns,
                                     class_names=['did not default', 'default'],
                                     mode='classification',
                                     random_state=random_state)

    if predict_proba_available:
        # Choose the j_th instance and use it to explain the model prediction
        exp = explainer.explain_instance(
            data_row=x_test.iloc[j],
            predict_fn=model.predict_proba)

        # Save the report
        exp.save_to_file(Path(lime_results_path, f'lime_report_obs_{j}_{model_name}.html'))

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

            # explain the true positive predicted instance
            explanation = explainer.explain_instance(x_test.iloc[tp_idx], model.predict_proba)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_tp_pred_{tp_idx}_{model_name}.html'))

            # explain the true negative predicted instance
            explanation = explainer.explain_instance(x_test.iloc[tn_idx], model.predict_proba)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_tn_pred_{tn_idx}_{model_name}.html'))

            # explain the false positive predicted instance
            explanation = explainer.explain_instance(x_test.iloc[fp_idx], model.predict_proba)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_fp_pred_{fp_idx}_{model_name}.html'))

            # explain the false negative predicted instance
            explanation = explainer.explain_instance(x_test.iloc[fn_idx], model.predict_proba)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_fn_pred_{fn_idx}_{model_name}.html'))

    else:
        # Choose the j_th instance and use it to explain the model prediction
        exp = explainer.explain_instance(
            data_row=x_test.iloc[j],
            predict_fn=predict_use_keras,
            top_labels=1
        )

        # Save the report
        exp.save_to_file(Path(lime_results_path, f'lime_report_obs_{j}_{model_name}.html'))

        if with_wrong_prediction_analysis:
            # Analyze wrong predictions
            y_pred = (model.predict(x_test) > 0.5).astype(int)
            my_list = map(lambda x: x[0], y_pred)
            y_pred = pd.Series(my_list)

            tp_idxs = np.where((y_test == 1) & (y_pred == 1))[0]
            tn_idxs = np.where((y_test == 0) & (y_pred == 0))[0]
            fp_idxs = np.where((y_test == 0) & (y_pred == 1))[0]
            fn_idxs = np.where((y_test == 1) & (y_pred == 0))[0]

            # choose a random index
            tp_idx = random.choice(tp_idxs)
            tn_idx = random.choice(tn_idxs)
            fp_idx = random.choice(fp_idxs)
            fn_idx = random.choice(fn_idxs)

            # explain the true positive predicted instance
            explanation = explainer.explain_instance(x_test.iloc[tp_idx], predict_use_keras, top_labels=1)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_tp_pred_{tp_idx}_{model_name}.html'))

            # explain the true negative predicted instance
            explanation = explainer.explain_instance(x_test.iloc[tn_idx], predict_use_keras, top_labels=1)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_tn_pred_{tn_idx}_{model_name}.html'))

            # explain the false positive predicted instance
            explanation = explainer.explain_instance(x_test.iloc[fp_idx], predict_use_keras, top_labels=1)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_fp_pred_{fp_idx}_{model_name}.html'))

            # explain the false negative predicted instance
            explanation = explainer.explain_instance(x_test.iloc[fn_idx], predict_use_keras, top_labels=1)

            # save the html file
            explanation.save_to_file(Path(lime_results_path, f'lime_report_fn_pred_{fn_idx}_{model_name}.html'))


def shap_explanation(training: pd.DataFrame, testing: pd.DataFrame, target: str, model: ...,
                     explainer_type: Literal["tree", "kernel_cnn", "kernel_svc"], model_name: str, j: int = 0) -> None:
    """
    This function carry out a Local Model-agnostic Explanation of a pre-trained model using the shap library:
    https://shap.readthedocs.io/en/latest/index.html
    @param training: pd.DataFrame: the training dataset.
    @param testing: pd.DataFrame: the testing dataset.
    @param target: str: the target column's name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param explainer_type: str: a string containing which type of algorithm our model is.
    @param model_name: str: the name of out model, to save appropriately the results.
    @param j: int: index of the data our model will use for the prediction.
    """

    # Splitting the data:
    x_train = training.drop(target, axis=1)
    x_test = testing.drop(target, axis=1)

    if explainer_type == "tree":
        # data to train explainer on
        background = maskers.Independent(x_train)

        # y_pred = model.predict(x_test)

        # initialize the shapley values:
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(x_test.iloc[j, :], check_additivity=False)

        shap.waterfall_plot(shap.Explanation(values=shap_values,
                                             base_values=explainer.expected_value,
                                             data=x_test.iloc[j],
                                             feature_names=x_test.columns.tolist()), show=False)

        # give more space for the y-axis:
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.2)
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.2)
        # increase the size of the plot:
        plt.gcf().set_size_inches(10, 5)
        plt.savefig(Path(shap_results_path, f'shap_waterfall_obs_{j}_{model_name}.png'))
        plt.close()

        # force plot of the observation
        shap.force_plot(explainer.expected_value,
                        shap_values,
                        x_test.values[j, :],
                        feature_names=x_test.columns,
                        matplotlib=True, show=False)

        plt.gcf().set_size_inches(30, 5)
        # Save the plot
        plt.savefig(Path(shap_results_path, f'shap_force_plot_obs_{j}_{model_name}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    elif explainer_type == "kernel_cnn":
        # initialize the shapley values:
        explainer = shap.KernelExplainer(model, x_train.iloc[:50, :])
        shap_values = explainer.shap_values(x_test.iloc[j, :])

        # force plot of the observation
        shap.force_plot(explainer.expected_value[0],
                        shap_values[0],
                        features=x_test.columns,
                        matplotlib=True, show=False)

        plt.savefig(Path(shap_results_path, f'shap_force_plot_obs_{j}_{model_name}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Decision plot of the observation
        shap.decision_plot(explainer.expected_value[0],
                           shap_values[0],
                           features=x_test.iloc[0, :],
                           feature_names=x_test.columns.tolist(),
                           show=False)

        # increase the size of the plot:
        plt.gcf().set_size_inches(10, 15)
        plt.savefig(Path(shap_results_path, f'shap_decision_plot_obs_{j}_{model_name}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Waterfall plot
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
                                               shap_values[0],
                                               feature_names=x_test.columns,
                                               show=False)
        # increase the size of the plot:
        plt.gcf().set_size_inches(10, 5)
        plt.savefig(Path(shap_results_path, f'shap_waterfall_plot_obs_{j}_{model_name}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    elif explainer_type == "kernel_svc":
        # initialize the shapley values:
        explainer = shap.KernelExplainer(model, x_train.iloc[:50, :], link="logit")
        shap_values = explainer.shap_values(x_test.iloc[j, :])

        # Force plot
        shap.force_plot(explainer.expected_value[1],
                        shap_values[1],
                        features=x_test.columns,
                        matplotlib=True, show=False)

        # plt.gcf().set_size_inches(15, 10)
        plt.savefig(Path(shap_results_path, f'shap_force_plot_obs_{j}_{model_name}.png'))
        plt.close()

        # Decision plot
        shap.decision_plot(explainer.expected_value[1],
                           shap_values[1],
                           features=x_test.iloc[j, :],
                           feature_names=x_test.columns.tolist(),
                           show=False)

        # increase the size of the plot:
        plt.gcf().set_size_inches(10, 15)
        plt.savefig(Path(shap_results_path, f'shap_decision_plot_obs_{j}_{model_name}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Waterfall plot
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                               shap_values[1],
                                               feature_names=x_test.columns,
                                               show=False)

        # increase the size of the plot:
        plt.gcf().set_size_inches(10, 5)
        plt.savefig(Path(shap_results_path, f'shap_waterfall_plot_obs_{j}_{model_name}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()


@measure_time
def lime_and_shap_main() -> None:
    """
    Main function to execute LIME and SHAP explainability methods on the trained models.
    """
    # load the data:
    training = pd.read_csv(final_train_under_csv_path)
    validation = pd.read_csv(final_val_under_csv_path)
    testing = pd.read_csv(final_test_under_csv_path)

    # concatenate the training and validation data:
    training = pd.concat([training, validation], axis=0)

    # Explaining surrogate model worst predictions
    bb = load_model(Path(global_surrogate_models_path, 'final_models/black_box_model.h5'))
    lr = pd.read_pickle(Path(global_surrogate_models_path, 'logistic_regression_surrogate_model.pkl'))
    rf = pd.read_pickle(Path(global_surrogate_models_path, 'random_forest_surrogate_model.pkl'))

    pred_logreg = pd.read_csv(Path(global_surrogate_results_path, "predictions_with_biggest_logreg_difference.csv"))
    pred_rf = pd.read_csv(Path(global_surrogate_results_path, "predictions_with_biggest_rf_difference.csv"))

    # Extracting the indexes of the worst prediction of our two surrogate models
    worst_surrogate_obs_logreg = [i for i in pred_logreg.iloc[:3, 0]]
    worst_surrogate_obs_rf = [i for i in pred_rf.iloc[:3, 0]]
    worst_obs = worst_surrogate_obs_logreg + worst_surrogate_obs_rf

    for o in worst_surrogate_obs_logreg:
        lime_explanation(training, testing, 'default', bb, 'black_box', o)
        lime_explanation(training, testing, 'default', lr, 'surrogate_log_reg', o)

    for o in worst_surrogate_obs_rf:
        lime_explanation(training, testing, 'default', bb, 'black_box', o)
        lime_explanation(training, testing, 'default', rf, 'surrogate_random_f', o)

    # load the pickled models:
    cnn_model = pd.read_pickle(Path(final_models_path, 'cnn_model.pkl'))
    gb_model = pd.read_pickle(Path(final_models_path, 'gradient_boosting_model.pkl'))
    svc_model = pd.read_pickle(Path(final_models_path, 'support_vector_machine_model.pkl'))

    models = [cnn_model, gb_model, svc_model]

    # train the model:
    for model in models:
        for idx in worst_obs:
            lime_explanation(training=training, testing=testing, target="default", j=idx,
                             model=model, model_name=model.__class__.__name__.lower())
            if model == cnn_model:
                shap_explanation(training=training, testing=testing, target="default", j=idx,
                                 model=model, model_name=model.__class__.__name__.lower(), explainer_type="kernel_cnn")

            elif model == gb_model:
                shap_explanation(training=training, testing=testing, target="default", j=idx,
                                 model=model, model_name=model.__class__.__name__.lower(), explainer_type="tree")

            elif model == svc_model:
                shap_explanation(training=training, testing=testing, target="default", j=idx,
                                 model=model.predict_proba, model_name=model.__class__.__name__.lower(),
                                 explainer_type="kernel_svc")

            else:
                raise ValueError("Model type not supported, please check the model type.")


# Driver code
if __name__ == '__main__':
    lime_and_shap_main()

