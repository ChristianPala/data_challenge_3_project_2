# Library to explain the results of the black box neural network model with a global surrogate:
import os
import warnings
# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from keras import Model
from keras.models import load_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Global variables:
from config import final_neural_network_path, global_surrogate_results_path, \
    final_training_csv_path, final_validation_csv_path, final_testing_csv_path

# Plotting:
import matplotlib.pyplot as plt

# Type hints:
from typing import Tuple

# Tensorflow appropriate compiler flags ignored:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ConvergenceWarning)


def load_data() -> tuple:
    """
    This function loads the data.
    :return: tuple: x_train, y_train, x_test, y_test
    """
    training = pd.read_csv(final_training_csv_path)
    validation = pd.read_csv(final_validation_csv_path)
    testing = pd.read_csv(final_testing_csv_path)
    x_train = training.drop(columns=['default'])
    y_train = training['default']
    x_val = validation.drop(columns=['default'])
    y_val = validation['default']
    # concatenate the training and validation data to have a larger training set:
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))
    x_test = testing.drop(columns=['default'])
    y_test = testing['default']

    return x_train, y_train, x_test, y_test


def import_black_box_model(x_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    This function imports the black box model and trains it.
    @param x_train: np.ndarray: the training data.
    @param y_train: np.ndarray: the target data.
    :return: keras.Model: the black box model.
    """
    black_box_model = load_model(final_neural_network_path)
    # reduced the number of epochs to 5 to keep to give the surrogate models a chance to learn:
    black_box_model.fit(x_train, y_train, epochs=5, verbose=0)
    return black_box_model


def generate_y_train(x_train: np.array, black_box_model: Model) -> np.array:
    """
    This function generates the target data for the surrogate model.
    @param x_train: np.array: the training data.
    @param black_box_model: keras.Model: the black box model.
    :return: np.array: the target data for the surrogate model.
    """
    y_train = black_box_model.predict(x_train, verbose=0)
    y_train = np.where(y_train > 0.5, 1, 0)
    return y_train


def create_and_train_surrogate_model(x_train: np.array, y_train: np.array) -> (LogisticRegression,
                                                                               RandomForestClassifier):
    """
    This function creates a surrogate model.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: LogisticRegression: the trained surrogate model.
    """
    surrogate_model_log = LogisticRegression()
    surrogate_model_rf = RandomForestClassifier()
    surrogate_model_log.fit(x_train, y_train)
    surrogate_model_rf.fit(x_train, y_train)
    return surrogate_model_log, surrogate_model_rf


def tune_surrogate_models_for_black_box_model(black_box_model: Model) \
        -> (LogisticRegression, RandomForestClassifier):
    """
    This function tunes the surrogate models.
    @param black_box_model: keras.Model: the black box model.
    :return: LogisticRegression: the tuned surrogate model.
    """
    x_train, y_train, x_test, _ = load_data()
    # cast to numpy arrays:
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # generate the target data for the surrogate model:
    y_train = generate_y_train(x_train, black_box_model).ravel()

    # create and train the surrogate model:
    surrogate_model_log, surrogate_model_rf = create_and_train_surrogate_model(x_train, y_train)
    params_log = {'C': [10, 30, 50, 70, 100]}

    surrogate_model_log = HalvingRandomSearchCV(surrogate_model_log, params_log, cv=5, n_jobs=4, verbose=0,
                                                random_state=42).fit(x_train, y_train)
    params_rf = {'n_estimators': [200, 210, 220, 230, 240, 250, 260], 'max_depth': [12, 14, 15, 16, 18, 20]}
    surrogate_model_rf = HalvingRandomSearchCV(surrogate_model_rf, params_rf, cv=5, n_jobs=-1, verbose=0,
                                               random_state=42).fit(x_train, y_train)
    # get the best models, already fitted:
    surrogate_model_log = surrogate_model_log.best_estimator_
    surrogate_model_rf = surrogate_model_rf.best_estimator_

    # evaluate the surrogate model:
    y_test = black_box_model.predict(x_test, verbose=0)
    y_test = np.where(y_test > 0.5, 1, 0)
    y_pred_log = surrogate_model_log.predict(x_test)
    y_pred_rf = surrogate_model_rf.predict(x_test)

    # accuracy:
    accuracy_log = accuracy_score(y_test, y_pred_log)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    print(f'The accuracy of the logistic regression is: {accuracy_log}')
    print(f'The accuracy of the random forest is: {accuracy_rf}')

    return surrogate_model_log, surrogate_model_rf


def test_surrogate_model(x_test: np.array, black_box_model: Model,
                         surrogate_model: Tuple[LogisticRegression, RandomForestClassifier]) -> (float, float):
    """
    This function tests how close the surrogate model is to the black box model using
    R = 1 - (sum of squared errors of surrogate model / sum of squared errors of black box model)
    @param x_test: np.array: the test data.
    @param surrogate_model: LogisticRegression: the surrogate model.
    :return: r_squared: float: the r_squared value, the closer to 1 the better.
    """

    # Get the predictions of the black box model:
    y_pred_bb = black_box_model.predict(x_test)
    y_pred_bb = np.where(y_pred_bb > 0.5, 1, 0)
    y_pred_bb_mean = np.mean(y_pred_bb)
    # vectorize to have the same shape as y_pred_bb:
    y_pred_bb_mean = np.vectorize(lambda x: y_pred_bb_mean)(y_pred_bb)

    # Get the predictions of the surrogate model:
    y_pred_surrogate_log = surrogate_model[0].predict(x_test)
    y_pred_surrogate_rf = surrogate_model[1].predict(x_test)

    # Calculate SST, using sklearn for robustness:
    sum_of_squares_total = mean_squared_error(y_pred_bb, y_pred_bb_mean) * len(y_pred_bb)

    # Calculate SSE, using sklearn for robustness:
    sum_squared_errors_surrogate_log = mean_squared_error(y_pred_surrogate_log, y_pred_bb) * len(y_pred_bb)
    sum_squared_errors_surrogate_rf = mean_squared_error(y_pred_surrogate_rf, y_pred_bb) * len(y_pred_bb)

    # Calculate the r_squared value:
    r_squared_log = 1 - (sum_squared_errors_surrogate_log / sum_of_squares_total)
    r_squared_rf = 1 - (sum_squared_errors_surrogate_rf / sum_of_squares_total)

    return r_squared_log, r_squared_rf


def interpret_surrogate_model(surrogate_models: tuple[LogisticRegression, RandomForestClassifier], x_test: np.array) \
        -> None:
    """
    This function interprets the surrogate model.
    @param surrogate_models: LogisticRegression and RandomForestClassifier: the surrogate models.
    @param x_test: np.array: the test data.
    :return: pd.DataFrame: the interpretation of the surrogate model.
    """
    surrogate_model_log = surrogate_models[0]
    surrogate_model_rf = surrogate_models[1]
    if surrogate_model_log:
        # Get the coefficients:
        coefficients = surrogate_model_log.coef_
        intercept = surrogate_model_log.intercept_
        # Create the dataframe:
        df = pd.DataFrame(data=coefficients, columns=x_test.columns)
        df['intercept'] = intercept
        df.to_csv(Path(global_surrogate_results_path, 'logistic_surrogate_model_interpretation.csv'))

    elif surrogate_model_rf:
        # Get the feature importance:
        feature_importance = surrogate_model_rf.feature_importances_
        # Create the dataframe:
        df = pd.DataFrame(data=feature_importance, index=x_test.columns, columns=['feature_importance'])
        df.to_csv(Path(global_surrogate_results_path, 'random_forest_surrogate_model_interpretation.csv'))


def print_results(r_squared: (float, float)) -> None:
    """
    This function prints and saves the results of the global surrogate model analysis.
    @param r_squared: (float, float): the r_squared value of the surrogate models.
    :return: None
    """
    print(f'The R_squared value for the logistic regression is: {r_squared[0]}')
    print(f'The R_squared value for the random forest is: {r_squared[1]}')


def save_model_predictions(black_box_model: Model, surrogate_models: tuple[LogisticRegression, RandomForestClassifier],
                           x_test: np.array) -> None:
    """
    This function saves the predictions of the black box model and the surrogate models.
    @param black_box_model: Model: the black box model.
    @param surrogate_models: tuple[LogisticRegression, RandomForestClassifier]: the surrogate models.
    @param x_test: np.array: the test data.
    :return: None
    """
    # Get the predictions of the black box model:
    y_pred_bb = black_box_model.predict(x_test)
    y_pred_bb = np.where(y_pred_bb > 0.5, 1, 0)
    # Get the predictions of the surrogate model:
    y_pred_surrogate_log = surrogate_models[0].predict(x_test)
    y_pred_surrogate_rf = surrogate_models[1].predict(x_test)
    # Create the dataframe:
    df = pd.DataFrame(data=y_pred_bb, columns=['black_box'])
    df['surrogate_log'] = y_pred_surrogate_log
    df['surrogate_rf'] = y_pred_surrogate_rf
    df.to_csv(Path(global_surrogate_results_path, 'predictions.csv'))


def plot_feature_importance(surrogate_models: tuple[LogisticRegression, RandomForestClassifier],
                            x_test: pd.DataFrame) -> None:
    """
    Plots the feature importance of the surrogate models.
    @param surrogate_models: tuple[LogisticRegression, RandomForestClassifier]: the surrogate models.
    @param x_test: pd.DataFrame: the test data.
    :return: None
    """
    surrogate_model_log = surrogate_models[0]
    surrogate_model_rf = surrogate_models[1]
    if surrogate_model_log:
        # Get the coefficients:
        coefficients = surrogate_model_log.coef_
        # Create the dataframe:
        df = pd.DataFrame(data=coefficients, columns=x_test.columns)
        # Plot the feature importance:
        df.plot.bar()
        plt.title('Feature importance of the logistic regression surrogate model')
        plt.xlabel('Feature')
        plt.ylabel('Coefficient')
        plt.savefig(Path(global_surrogate_results_path, 'logistic_surrogate_model_feature_importance.png'))
        plt.show()

    elif surrogate_model_rf:
        # Get the feature importance:
        feature_importance = surrogate_model_rf.feature_importances_
        # Create the dataframe:
        df = pd.DataFrame(data=feature_importance, index=x_test.columns, columns=['feature_importance'])
        # Plot the feature importance:
        df.plot.bar()
        plt.title('Feature importance of the random forest surrogate model')
        plt.xlabel('Feature')
        plt.ylabel('Feature importance')
        plt.savefig(Path(global_surrogate_results_path, 'random_forest_surrogate_model_feature_importance.png'))
        plt.show()


def global_surrogate_main() -> None:
    """
    This function runs the global surrogate model analysis.
    :return: None
    """
    # Import the data:
    x_train, y_train, x_test, y_test = load_data()

    # Import the black box model:
    black_box_model = import_black_box_model(np.array(x_train), np.array(y_train))

    # Create and train the surrogate model:
    surrogate_models = tune_surrogate_models_for_black_box_model(black_box_model)

    # Test the surrogate model:
    r_squared = test_surrogate_model(x_test, black_box_model, surrogate_models)

    # Interpret the surrogate model:
    global_surrogate_results_path.mkdir(parents=True, exist_ok=True)
    interpret_surrogate_model(surrogate_models, x_test)

    # Print and save the results:
    print_results(r_squared)

    # Save the predictions:
    save_model_predictions(black_box_model, surrogate_models, x_test)

    # Plot the feature importance:
    plot_feature_importance(surrogate_models, x_test)


if __name__ == '__main__':
    global_surrogate_main()
