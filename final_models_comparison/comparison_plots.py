# Library to compare the results of the final models:
# Data manipulation:
import pandas as pd
from pathlib import Path
import numpy as np
import os
# Plotting:
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Model
from sklearn.ensemble import GradientBoostingClassifier
# Metrics:
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    roc_curve
from sklearn.svm import SVC

from auxiliary.method_timer import measure_time
# Global variables:
from config import final_models_path, final_test_csv_path, final_models_comparison_path

# tensorflow loggings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure that the directory exists:
final_models_comparison_path.mkdir(parents=True, exist_ok=True)


def load_test_data() -> (pd.DataFrame, pd.DataFrame):
    """
    Function to load the test data.
    :return: the test data
    """
    # Load the test data:
    test_df = pd.read_csv(final_test_csv_path)
    x_test = test_df.drop("default", axis=1)
    y_test = test_df["default"]
    return x_test, y_test


def load_models() -> (GradientBoostingClassifier, Model, SVC):
    """
    Function to load the final, trained, models.
    :return: the final models, pre-trained.
    """
    # Load the trained models:
    gb_model = pd.read_pickle(Path(final_models_path, "gradient_boosting_model.pkl"))
    cnn_model = pd.read_pickle(Path(final_models_path, "cnn_model.pkl"))
    svm_model = pd.read_pickle(Path(final_models_path, "support_vector_machine_model.pkl"))
    return gb_model, cnn_model, svm_model


def save_confusion_matrix_plot(y_true: pd.DataFrame, y_pred: pd.DataFrame, model_name: str) -> None:
    """
    Function to save the confusion matrix plots.
    @param y_true: the true labels
    @param y_pred: the predicted labels
    @param model_name: the name of the model
    :return: None. Saves the confusion matrix plot.
    """
    # Create the confusion matrix:
    cm = confusion_matrix(y_true, y_pred)
    # Plot the confusion matrix:
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    path = Path(final_models_comparison_path, "confusion_matrix_plots")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path, f"{model_name}_confusion_matrix.png"))
    plt.close()


def barplot_accuracy_f1_recall_precision(y_true: pd.DataFrame, y_pred: pd.DataFrame, model_name: str):
    """
    Function to save the barplots of the accuracy, F1, recall and precision scores.
    @param y_true: the true labels
    @param y_pred: the predicted labels
    @param model_name: the name of the model
    """
    # Create the dataframe:
    df = pd.DataFrame({"accuracy": [accuracy_score(y_true, y_pred)],
                       "f1": [f1_score(y_true, y_pred)],
                       "recall": [recall_score(y_true, y_pred)],
                       "precision": [precision_score(y_true, y_pred)]})

    # Plot the barplot:
    plt.figure(figsize=(10, 10))
    sns.barplot(data=df)
    plt.title(f"{model_name} accuracy, F1, recall and precision scores")
    path = Path(final_models_comparison_path, "scores_barplots")
    path.mkdir(parents=True, exist_ok=True)
    plt.xlabel("Scores")
    plt.ylabel("Value")
    plt.savefig(Path(path, f"{model_name}_scores_barplot.png"))
    plt.close()


def roc_auc_curve(y_true: pd.DataFrame, y_pred_proba: pd.DataFrame, model_name: str) -> None:
    """
    Function to save the ROC AUC curve.
    @param y_true: the true labels
    @param y_pred_proba: the predicted labels, as probabilities
    @param model_name: the name of the model
    :return: None. Saves the ROC AUC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f"{model_name} ROC curve (area = {roc_auc_score(y_true, y_pred_proba):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} ROC AUC curve")
    plt.legend(loc="lower right")
    path = Path(final_models_comparison_path, "roc_auc_curves")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path, f"{model_name}_roc_auc_curve.png"))
    plt.close()


@measure_time
def final_comparisons_main():
    """
    Main function.
    """
    # Load the test data:
    x_test, y_test = load_test_data()
    # Load the models:
    gb_model, cnn_model, svm_model = load_models()
    # Predict the labels, categorical and probabilities:
    gb_pred = gb_model.predict(x_test)
    gp_pred_proba = gb_model.predict_proba(x_test)
    # keep only the most probable label:
    gp_pred_proba = gp_pred_proba[:, 1]
    cnn_pred_proba = cnn_model.predict(x_test)
    cnn_pred = np.where(cnn_pred_proba > 0.5, 1, 0).ravel()
    svm_pred = svm_model.predict(x_test)
    svm_pred_proba = svm_model.predict_proba(x_test)
    # keep only the most probable label:
    svm_pred_proba = svm_pred_proba[:, 1]
    # model names:
    model_names = ("gradient boosting classifier", "convoluted neural network", "support vector machine classifier")

    # Save the confusion matrix plots:
    save_confusion_matrix_plot(y_test, gb_pred, model_names[0])
    save_confusion_matrix_plot(y_test, cnn_pred, model_names[1])
    save_confusion_matrix_plot(y_test, svm_pred, model_names[2])
    # Save the barplots:
    barplot_accuracy_f1_recall_precision(y_test, gb_pred, model_names[0])
    barplot_accuracy_f1_recall_precision(y_test, cnn_pred, model_names[1])
    barplot_accuracy_f1_recall_precision(y_test, svm_pred, model_names[2])
    # Save the ROC AUC curves:
    roc_auc_curve(y_test, gp_pred_proba, model_names[0])
    roc_auc_curve(y_test, cnn_pred_proba, model_names[1])
    roc_auc_curve(y_test, svm_pred_proba, model_names[2])


# Driver code:
if __name__ == "__main__":
    final_comparisons_main()
