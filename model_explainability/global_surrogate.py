# Library to explain the results of the black box neural network model with a global surrogate:

# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from keras import Model
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.special import kl_div

# Global variables:
from config import black_box_model_path, balanced_datasets_path, global_surrogate_results_path


# Note this has to match the data used to train the black box model:
def load_data() -> tuple:
    """
    This function loads the data.
    :return: tuple: x_train, y_train, x_test, y_test
    """
    train = Path(balanced_datasets_path, "undersampled", "minmax_scaler_scaling_most_frequent_imputation", "train.csv")
    test = Path(balanced_datasets_path, "undersampled", "minmax_scaler_scaling_most_frequent_imputation", "test.csv")

    x_train = pd.read_csv(train).drop(columns=['default'])
    y_train = pd.read_csv(train)['default']
    x_test = pd.read_csv(test).drop(columns=['default'])
    y_test = pd.read_csv(test)['default']

    return x_train, y_train, x_test, y_test


def import_black_box_model() -> Model:
    """
    This function imports the black box model.
    :return: keras.Model: the black box model.
    """
    return load_model(black_box_model_path)


def generate_y_train(x_train: np.array, black_box_model: Model) -> np.array:
    """
    This function generates the target data for the surrogate model.
    @param x_train: np.array: the training data.
    @param black_box_model: keras.Model: the black box model.
    :return: np.array: the target data for the surrogate model.
    """
    y_train = black_box_model.predict(x_train)
    y_train = np.where(y_train > 0.5, 1, 0)
    return y_train


def create_and_train_surrogate_model(x_train: np.array, y_train: np.array) -> RandomForestClassifier:
    """
    This function creates a surrogate model.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: LogisticRegression: the trained surrogate model.
    """
    surrogate_model = RandomForestClassifier()
    surrogate_model.fit(x_train, y_train)
    return surrogate_model


def test_surrogate_model(x_test: np.array, black_box_model: Model,
                         surrogate_model: LogisticRegression) -> float:
    """
    This function tests how close the surrogate model is to the black box model using
    R = 1 - (sum of squared errors of surrogate model / sum of squared errors of black box model)
    @param x_test: np.array: the test data.
    @param surrogate_model: LogisticRegression: the surrogate model.
    :return: r_squared: float: the r_squared value, the closer to 1 the better.
    """

    # Get the predictions of the black box model:
    y_pred_bb = black_box_model.predict(x_test)
    y_pred_bb_mean = np.mean(y_pred_bb)
    # vectorize to have the same shape as y_pred_bb:
    y_pred_bb_mean = np.vectorize(lambda x: y_pred_bb_mean)(y_pred_bb)

    # Get the predictions of the surrogate model:
    y_pred_surrogate = surrogate_model.predict(x_test)

    # Calculate SST:
    sum_of_squares_total = mean_squared_error(y_pred_bb, y_pred_bb_mean) * len(y_pred_bb)

    # Calculate SSE:
    sum_squared_errors_surrogate = mean_squared_error(y_pred_bb, y_pred_surrogate) * len(y_pred_bb)

    # Calculate the r_squared value:
    r_squared = 1 - (sum_squared_errors_surrogate / sum_of_squares_total)

    return r_squared


def interpret_surrogate_model(surrogate_model: LogisticRegression, x_test: np.array) -> pd.DataFrame:
    """
    This function interprets the surrogate model.
    @param surrogate_model: LogisticRegression: the surrogate model.
    @param x_test: np.array: the test data.
    :return: pd.DataFrame: the interpretation of the surrogate model.
    """
    if isinstance(surrogate_model, LogisticRegression):
        # Get the coefficients:
        coefficients = surrogate_model.coef_
        intercept = surrogate_model.intercept_
        # Create the dataframe:
        df = pd.DataFrame(data=coefficients, columns=x_test.columns)
        df['intercept'] = intercept

    elif isinstance(surrogate_model, RandomForestClassifier):
        # Get the feature importance:
        feature_importance = surrogate_model.feature_importances_
        # Create the dataframe:
        df = pd.DataFrame(data=feature_importance, index=x_test.columns, columns=['feature_importance'])
    return df


def print_and_save_results(r_squared: float, df: pd.DataFrame, output_path: Path) -> None:
    """
    This function prints and saves the results of the global surrogate model analysis.
    @param r_squared: float: the r_squared value of the surrogate model.
    @param df: pd.DataFrame: the weights and intercepts of the surrogate model.
    @param output_path: Path: the path to save the results.
    :return: None
    """
    print(f'The R_squared value is: {r_squared}')
    print(df.transpose())
    df.to_csv(Path(output_path, 'global_surrogate_results.csv'))


def global_surrogate_main() -> None:
    """
    This function runs the global surrogate model analysis.
    :return: None
    """
    # Import the black box model:
    black_box_model = import_black_box_model()

    # Import the data:
    x_train, y_train, x_test, y_test = load_data()

    # Generate the target data for the surrogate model:
    y_train_surrogate = generate_y_train(x_train, black_box_model).ravel()

    # Create and train the surrogate model:
    surrogate_model = create_and_train_surrogate_model(x_train, y_train_surrogate)

    # Test the surrogate model:
    r_squared = test_surrogate_model(x_test, black_box_model, surrogate_model)

    # Interpret the surrogate model:
    df = interpret_surrogate_model(surrogate_model, x_test)

    # Print and save the results:
    global_surrogate_results_path.mkdir(parents=True, exist_ok=True)
    print_and_save_results(r_squared, df, global_surrogate_results_path)


if __name__ == '__main__':
    global_surrogate_main()
