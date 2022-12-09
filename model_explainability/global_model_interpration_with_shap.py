# use the shap package to explain the three models, see the following link for the library
# documentation: https://shap.readthedocs.io/en/latest/index.html
# Libraries:
# Data manipulation:
from pathlib import Path
import os
import pandas as pd
# Explaining:
import shap
# Plotting:
import matplotlib.pyplot as plt
# Ignore deprecated warnings:
import warnings

from keras import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Global variables:
from config import final_models_path, final_test_csv_path, shap_results_path, final_train_csv_path, final_val_csv_path

# Number of samples the kernel explainer will use to train and explain the model:
NR_SAMPLES = 100
# Tensorflow logging:
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_data() -> (pd.DataFrame, pd.DataFrame):
    """
    Function to load the data for the explainability analysis with the SHAP library.
    """
    # load the training data:
    train_df = pd.read_csv(final_train_csv_path)
    val_df = pd.read_csv(final_val_csv_path)
    # concatenate the training and validation data:
    train_df = pd.concat([train_df, val_df], axis=0)
    x_train = train_df.drop("default", axis=1)
    # Load the test data:
    test_df = pd.read_csv(final_test_csv_path)
    x_test = test_df.drop("default", axis=1)
    return x_train, x_test


def load_models() -> (GradientBoostingClassifier, Model, SVC):
    """
    Function to load the final, trained, models.
    @return: the final models
    """
    # Load the trained models:
    gb_model = pd.read_pickle(Path(final_models_path, "gradient_boosting_model.pkl"))
    cnn_model = pd.read_pickle(Path(final_models_path, "cnn_model.pkl"))
    svm_model = pd.read_pickle(Path(final_models_path, "support_vector_machine_model.pkl"))
    return gb_model, cnn_model, svm_model


def create_explainers(x_train: pd.DataFrame,
                      gb_model: GradientBoostingClassifier, cnn_model: Model, svc_model: SVC) \
        -> (shap.KernelExplainer, shap.KernelExplainer, shap.KernelExplainer):
    """
    Function to create the SHAP explainer objects for the three models.
    @param x_train: the training data
    @param gb_model: the gradient boosting model
    @param cnn_model: the convoluted neural network model
    @param svc_model: the support vector machine model
    :return: the explainer objects
    """

    # sample a portion of the training data to train the kernel explainer, for time efficiency:
    x_train_sample = shap.sample(x_train, nsamples=NR_SAMPLES, random_state=42)

    # Create the explainers:
    gb_explainer = shap.TreeExplainer(gb_model)
    cnn_explainer = shap.KernelExplainer(cnn_model.predict, x_train_sample)
    svm_explainer = shap.KernelExplainer(svc_model.predict_proba, x_train_sample)
    return gb_explainer, cnn_explainer, svm_explainer


def save_shap_feature_importance(model: str, shap_values: list, x_test: pd.DataFrame) -> None:
    """
    Function to plot the feature importance for the given model.
    @param model: the model name
    @param shap_values: the shap values for the model
    @param x_test: the test data
    """
    shap.summary_plot(shap_values, x_test, plot_type="bar", feature_names=x_test.columns, show=False)
    plt.title(f"{model} Feature Importance")
    # give the title a bit more space:
    plt.subplots_adjust(top=0.9)
    plt.savefig(Path(shap_results_path, f"{model.lower().replace(' ', '_')}_feature_importance.png"))


def save_summary_plot(model: str, shap_values: list, x_test: pd.DataFrame) -> None:
    """
    Function to save the summary plot for the given model and shap values.
    @param model: the model name
    @param shap_values: the shap values for the model
    @param x_test: the test data
    """
    shap.summary_plot(shap_values, x_test, show=False)
    plt.title(f"{model} Summary Plot")
    # give the title a bit more space:
    plt.subplots_adjust(top=0.9)
    plt.savefig(Path(shap_results_path, f"{model.lower().replace(' ', '_')}_summary_plot.png"))


def global_shap_main() -> None:
    """
    Function to globally explain the final models with the SHAP library.
    """
    # Load the data:
    x_train, x_test = load_data()
    # Load the models:
    gb_model, cnn_model, svm_model = load_models()
    # Create the explainers:
    gb_explainer, cnn_explainer, svm_explainer = create_explainers(x_train, gb_model, cnn_model, svm_model)
    # Sample a portion of the test data to explain the models, for time efficiency:
    x_test_sample = shap.sample(x_test, nsamples=NR_SAMPLES, random_state=42)
    # Explain the models:
    gb_shap_values = gb_explainer.shap_values(x_test)
    cnn_shap_values = cnn_explainer.shap_values(x_test_sample.values)
    svm_shap_values = svm_explainer.shap_values(x_test_sample.values)

    # save the plots:
    save_shap_feature_importance("Gradient Boosting", gb_shap_values, x_test)
    save_shap_feature_importance("Convoluted Neural Network", cnn_shap_values, x_test_sample)
    save_shap_feature_importance("Support Vector Machine", svm_shap_values, x_test_sample)


# Driver Code:
if __name__ == '__main__':
    global_shap_main()
