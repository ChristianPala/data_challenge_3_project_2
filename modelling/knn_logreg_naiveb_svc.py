# auxiliary library to create knn, logistic regression and svm models for the second data challenge 3 project.

# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import shutil

import pandas as pd
# Modelling:
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    classification_report, confusion_matrix
from modelling.train_test_validation_split import split_data

# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
scaled_datasets_path: Path = Path("..", "data", "scaled_datasets")
results_path: Path = Path("..", "results")


# Functions:
def create_knn_model() -> KNeighborsClassifier:
    """
    This function creates a knn model.
    :return: KNeighborsClassifier: the model.
    """
    model = KNeighborsClassifier(n_neighbors=5)
    return model


def create_logreg_model() -> LogisticRegression:
    """
    This function creates a logistic regression model.
    :return: LogisticRegression: the model.
    """
    model = LogisticRegression(random_state=42)
    return model


def create_svm_model() -> SVC:
    """
    This function creates a svm model.
    :return: SVC: the model.
    """
    model = SVC(random_state=42)
    return model


def create_naive_bayes_model() -> GaussianNB:
    """
    This function creates a naive bayes model.
    :return: GaussianNB: the model.
    """
    model = GaussianNB()
    return model


def fit_model(model, x_train, y_train) -> BaseEstimator:
    """
    This function fits the model to the training data.
    @param model: model: the model to be fitted.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: Fitted model object.
    """
    model.fit(x_train, y_train)
    return model


def predict(model, x_test) -> np.array:
    """
    This function predicts the target data for the test data.
    @param model: model: the model to be used for the prediction.
    @param x_test: np.array: the test data.
    :return: np.array: the predicted target data.
    """
    y_pred = model.predict(x_test)
    return y_pred


def evaluate_model(y_test: np.array, y_pred: np.array) -> dict[str, float]:
    """
    This function evaluates the model's performance.
    @param y_test: np.array: the target values for the test data.
    @param y_pred: np.array: the predicted target values.
    :return: Dictionary with the metrics.
    """
    y_pred = np.where(y_pred > 0.5, 1, 0)
    df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

    return {
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'confusion_matrix': np.array2string(confusion_matrix(y_test, y_pred, normalize='true'), precision=2,
                                            separator=', '),
        'classification_report': df

    }


def save_evaluation_results(evaluation_results: dict, model_type: str, name_addition: str = None,
                            path_addition: Path = None) -> None:
    """
    This function saves the evaluation results to a file.
    @param evaluation_results: dict: the dictionary with the evaluation results.
    @param model_type: str: the type of the model.
    @param name_addition: str: default = None: the name addition to the file name to save the results.
    @param path_addition: Path: default = None: the path addition to the path to save the results.
    :return: None. Saves the results to a file in the results' folder.
    """

    other_models_results_path: Path = Path(results_path, "other_models")
    other_models_results_path.mkdir(parents=True, exist_ok=True)

    if path_addition:
        other_models_results_path = Path(other_models_results_path, path_addition)

    # write the results to a file:
    with open(other_models_results_path / f'{model_type}_base_evaluation_results_{name_addition}.txt', 'w') as f:
        for key, value in evaluation_results.items():
            if key == 'confusion_matrix':
                f.write(f'{key}\n {value}\n')
            elif key == 'classification_report':
                # empty line
                f.write('\n')
                value.to_csv(f, mode='a', header=True, sep='\t')
            else:
                f.write(f'{key}: {value}\n')


@measure_time
def other_models_main(additonal_subfolder_path: str = None) -> None:
    """
    This function creates the knn, logistic regression and svm base models and evaluates them
    on the project's dataset
    """

    # get all the csv files in the missing_values_handled folder
    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))

    # clean the neural networks results' directory:
    other_models_results_path: Path = Path(results_path, "other_models")
    if other_models_results_path.exists() and other_models_results_path.is_dir():
        shutil.rmtree(other_models_results_path, ignore_errors=True)
    other_models_results_path.mkdir(parents=True, exist_ok=True)

    # create the models:
    knn_model: KNeighborsClassifier = create_knn_model()
    logreg_model: LogisticRegression = create_logreg_model()
    svm_model: SVC = create_svm_model()
    naive_bayes_model: GaussianNB = create_naive_bayes_model()

    # loop through the csv files:
    for csv_file in csv_files:
        # read the csv file:
        df = pd.read_csv(csv_file, index_col=0)
        # split the data into train and test:
        x_train, x_test, y_train, y_test = split_data(df, 'default')

        # fit the models:
        knn_model = fit_model(knn_model, x_train, y_train)
        logreg_model = fit_model(logreg_model, x_train, y_train)
        naive_bayes_model = fit_model(naive_bayes_model, x_train, y_train)
        svm_model = fit_model(svm_model, x_train, y_train)


        # predict the target values:
        knn_y_pred = predict(knn_model, x_test)
        logreg_y_pred = predict(logreg_model, x_test)
        naive_bayes_y_pred = predict(naive_bayes_model, x_test)
        svm_y_pred = predict(svm_model, x_test)

        # evaluate the models:
        knn_evaluation_results = evaluate_model(y_test, knn_y_pred)
        logreg_evaluation_results = evaluate_model(y_test, logreg_y_pred)
        naive_bayes_evaluation_results = evaluate_model(y_test, naive_bayes_y_pred)
        svm_evaluation_results = evaluate_model(y_test, svm_y_pred)

        # save the results:
        if additonal_subfolder_path is not None:
            save_evaluation_results(knn_evaluation_results, 'knn', additonal_subfolder_path)
            save_evaluation_results(logreg_evaluation_results, 'logreg', additonal_subfolder_path)
            save_evaluation_results(naive_bayes_evaluation_results, 'naive_bayes', additonal_subfolder_path)
            save_evaluation_results(svm_evaluation_results, 'svm', additonal_subfolder_path)

        save_evaluation_results(knn_evaluation_results, 'knn', csv_file.stem)
        save_evaluation_results(logreg_evaluation_results, 'logreg', csv_file.stem)
        save_evaluation_results(naive_bayes_evaluation_results, 'naive_bayes', csv_file.stem)
        save_evaluation_results(svm_evaluation_results, 'svm', csv_file.stem)


if __name__ == '__main__':
    other_models_main()

