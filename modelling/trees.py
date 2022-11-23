# modelling with decision trees, random forests and gradient boosting

# Libraries
import pandas as pd
import numpy as np

# Data manipulation
from pathlib import Path

# Modelling
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from train_test_validation_split import split_data

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report

# Global variables
logs_path = Path('..', 'logs')


def generate_tree_model(model_type: str) -> BaseEstimator:
    """
    This function generates one of the tree models.
    @param model_type: str: the type of the model to be generated. Supported:
     - 'decision_tree',
     - 'random_forest',
     - 'gradient_boosting'
     - 'xgboost'.
    :return: Model object.
    """
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(random_state=42)
    elif model_type == 'random_forest':
        return RandomForestClassifier(random_state=42)
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=42)
    elif model_type == 'xgboost':
        return XGBClassifier(random_state=42)
    else:
        raise ValueError('The model type is not supported.')


def fit_model(model: BaseEstimator, x_train: np.array, y_train: np.array) -> BaseEstimator:
    """
    This function fits the model to the training data.
    @param model: BaseEstimator: the model to be fitted.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: Fitted model object.
    """
    model.fit(x_train, y_train)
    return model


def predict_model(model: BaseEstimator, x_test: np.array) -> np.array:
    """
    This function predicts the target values for the test data.
    @param model: BaseEstimator: the model to be used for prediction.
    @param x_test: np.array: the test data.
    :return: Predicted target values as np.array.
    """
    return model.predict(x_test)


def evaluate_model(y_test: np.array, y_pred: np.array) -> dict:
    """
    This function evaluates the model's performance.
    @param y_test: np.array: the target values for the test data.
    @param y_pred: np.array: the predicted target values.
    :return: Dictionary with the metrics.
    """
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }


def save_evaluation_results(evaluation_results: dict, model_type: str) -> None:
    """
    This function saves the evaluation results to a file.
    @param evaluation_results: dict: the dictionary with the evaluation results.
    @param model_type: str: the type of the model.
    :return: None. Saves the results to a file in the logs folder.
    """
    with open(logs_path / f'{model_type}_base_evaluation_results.txt', 'w') as f:
        for key, value in evaluation_results.items():
            f.write(f'{key}: {value}\n')


def main() -> None:
    """
    Main function for the baseline results on decision trees, random forests, gradient boosting and
    xgboost classifiers on the project 2 dataset.
    @return: None, saves the results in the logs folder.
    """
    df = pd.read_csv(Path('..', 'data', 'project_2_dataset.csv'))
    x_train, x_test, y_train, y_test = split_data(df, 'default')
    model_types = ['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost']
    models = [generate_tree_model(model_type) for model_type in model_types]
    for model, model_type in zip(models, model_types):
        fitted_model = fit_model(model, x_train, y_train)
        y_pred = predict_model(fitted_model, x_test)
        evaluation_results = evaluate_model(y_test, y_pred)
        save_evaluation_results(evaluation_results, model_type)


if __name__ == '__main__':
    main()
