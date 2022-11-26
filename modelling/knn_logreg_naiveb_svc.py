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
from modelling.model_evaluator import save_evaluation_results, evaluate_model
from modelling.train_test_validation_split import split_data
# Timing:
from auxiliary.method_timer import measure_time
from tqdm import tqdm
# Global variables:
from config import scaled_datasets_path, other_models_results_path


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


def create_svc_model() -> SVC:
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


def predict_model(model, x_test) -> np.array:
    """
    This function predicts the target data for the test data.
    @param model: model: the model to be used for the prediction.
    @param x_test: np.array: the test data.
    :return: np.array: the predicted target data.
    """
    y_pred = model.predict(x_test)
    return y_pred


@measure_time
def other_models_main() -> None:
    """
    This function creates the knn, logistic regression and svm base models and evaluates them
    on the project's dataset
    """
    if not scaled_datasets_path.exists():
        raise FileNotFoundError(f'The path {scaled_datasets_path} does not exist. '
                                f'Please create the scaled datasets first.')

    # get all the csv files in the missing_values_handled folder
    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))

    # clean the results folder
    if other_models_results_path.exists() and other_models_results_path.is_dir():
        shutil.rmtree(other_models_results_path, ignore_errors=True)
    other_models_results_path.mkdir(parents=True, exist_ok=True)

    # create the models:
    knn_model: KNeighborsClassifier = create_knn_model()
    logreg_model: LogisticRegression = create_logreg_model()
    svc_model: SVC = create_svc_model()
    naive_bayes_model: GaussianNB = create_naive_bayes_model()

    # loop through the csv files:
    for csv_file in tqdm(csv_files, desc='Other models', unit='file', total=len(csv_files)):
        # read the csv file:
        df = pd.read_csv(csv_file, index_col=0)
        # split the data into train and test:
        x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)

        # fit the models:
        knn_model_fit = fit_model(knn_model, x_train, y_train)
        logreg_model_fit = fit_model(logreg_model, x_train, y_train)
        naive_bayes_model_fit = fit_model(naive_bayes_model, x_train, y_train)
        svc_model_fit = fit_model(svc_model, x_train, y_train)

        # predict the target values:
        knn_y_pred = predict_model(knn_model_fit, x_val)
        logreg_y_pred = predict_model(logreg_model_fit, x_val)
        naive_bayes_y_pred = predict_model(naive_bayes_model_fit, x_val)
        svm_y_pred = predict_model(svc_model_fit, x_val)

        # evaluate the models:
        knn_evaluation_results = evaluate_model(y_val, knn_y_pred)
        logreg_evaluation_results = evaluate_model(y_val, logreg_y_pred)
        naive_bayes_evaluation_results = evaluate_model(y_val, naive_bayes_y_pred)
        svc_evaluation_results = evaluate_model(y_val, svm_y_pred)

        # save the results:
        save_evaluation_results(knn_evaluation_results, 'knn', other_models_results_path, csv_file.stem)
        save_evaluation_results(logreg_evaluation_results, 'logreg', other_models_results_path, csv_file.stem)
        save_evaluation_results(naive_bayes_evaluation_results, 'naive_bayes', other_models_results_path, csv_file.stem)
        save_evaluation_results(svc_evaluation_results, 'svc', other_models_results_path, csv_file.stem)


if __name__ == '__main__':
    other_models_main()

