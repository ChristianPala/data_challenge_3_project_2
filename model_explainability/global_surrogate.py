# Library to explain the results of the black box neural network model with a global surrogate:

# Libraries:
# Data manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from keras import Model
from keras.models import load_model
from sklearn.linear_model import LogisticRegression

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

    x_train = pd.read_csv(train, index_col=0).drop(columns=['default'])
    y_train = pd.read_csv(train, index_col=0)['default']
    x_test = pd.read_csv(test, index_col=0).drop(columns=['default'])
    y_test = pd.read_csv(test, index_col=0)['default']

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
    return y_train


def create_surrogate_model(x_train: np.array, y_train: np.array) -> LogisticRegression:
    """
    This function creates a surrogate model.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: LogisticRegression: the trained surrogate model.
    """
    surrogate_model = LogisticRegression()
    surrogate_model.fit(x_train, y_train)
    return surrogate_model


def test_surrogate_model(x_test: np.array, y_test: np.array, black_box_model: Model,
                         surrogate_model: LogisticRegression) -> float:
    """
    This function tests how close the surrogate model is to the black box model using
    R = 1 - (sum of squared errors of surrogate model / sum of squared errors of black box model)
    @param x_test: np.array: the test data.
    @param y_test: np.array: the target data.
    @param surrogate_model: LogisticRegression: the surrogate model.
    :return: r_squared: float: the r_squared value, the closer to 1 the better.
    """

    # Calculate the sum of squared errors of the black box model:
    y_pred_bb = black_box_model.predict(x_test)
    sse_bb = np.sum((y_pred_bb - y_test) ** 2)

    # Calculate the sum of squared errors of the surrogate model:
    y_pred_surrogate = surrogate_model.predict(x_test)
    sse_surrogate = np.sum((y_pred_surrogate - y_test) ** 2)

    # Calculate the r_squared value:
    r_squared = 1 - (sse_surrogate / sse_bb)
    return r_squared


def interpret_surrogate_model(surrogate_model: LogisticRegression, x_test: np.array) -> pd.DataFrame:
    """
    This function interprets the surrogate model.
    @param surrogate_model: LogisticRegression: the surrogate model.
    @param x_test: np.array: the test data.
    :return: pd.DataFrame: the interpretation of the surrogate model.
    """
    # Get the coefficients:
    coefficients = surrogate_model.coef_
    intercept = surrogate_model.intercept_

    # Create the dataframe:
    df = pd.DataFrame(data=coefficients, columns=x_test.columns)
    df['intercept'] = intercept
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
    df.to_csv(output_path)


def global_surrogate_main() -> None:
    """
    This function runs the global surrogate model analysis.
    :return: None
    """
    # Import the black box model:
    black_box_model = import_black_box_model()

    # Import the data:
    x_train, y_train, x_test, y_test = load_data()

    # Create and train the surrogate model:
    surrogate_model = create_surrogate_model(x_train, y_train)

    # Test the surrogate model:
    r_squared = test_surrogate_model(x_test, y_test, black_box_model, surrogate_model)

    # Interpret the surrogate model:
    df = interpret_surrogate_model(surrogate_model, x_test)

    # Print and save the results:
    output_path = Path(global_surrogate_results_path, "global_surrogate_model_results.csv")
    print_and_save_results(r_squared, df, output_path)


if __name__ == '__main__':
    global_surrogate_main()
