# Library to scale and normalize the data.


# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Scaling and normalization:
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, normalize

from auxiliary.method_timer import measure_time
# Modelling:
from modelling.train_test_validation_split import split_data
from preprocessing.skeweness_handling import skewness_main

# Global variables:
# Path to the data folder:
data_path = Path("..", "data")
# Path to the missing_values_handled folder:
missing_values_path = Path(data_path, "missing_values_handled")
missing_values_path.mkdir(exist_ok=True)
# path to the scaling data:
scaled_datasets_path = Path("..", "data", "scaled_datasets")
scaled_datasets_path.mkdir(parents=True, exist_ok=True)


def scale_data(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], method: str = "Standard") \
        -> pd.DataFrame:
    """
    Scale the data using the specified method, taken from Scikit-learn's preprocessing module:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    @param train: pd.DataFrame: the dataframe containing the training data.
    @param test: pd.DataFrame: the dataframe containing the test data.
    @param columns: list[str]: the list of columns to scale.
    @param method: the method to use for scaling. Supported methods are: "Standard", "MinMax", "Robust".
    @return: the scaled dataset.
    """
    if method == "Robust":
        scaler = RobustScaler()
    elif method == "MinMax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # fit the scaler on the training data:
    scaler.fit(train[columns])
    # scale the training data:
    train[columns] = scaler.transform(train[columns])

    # scale the test data:
    test[columns] = scaler.transform(test[columns])

    # merge the training and test data:
    dataframe: pd.DataFrame = pd.concat([train, test], axis=0)

    return dataframe


def normalize_data(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> (pd.DataFrame, pd.DataFrame):
    """
    Normalize the data using Scikit-learn's normalize method:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    @param train: the dataframe containing the training data.
    @param test: the dataframe containing the test data.
    @param columns: the list of columns to normalize.
    @return: the normalized dataset
    """

    # normalize the training data:
    train[columns] = normalize(train[columns])
    # normalize the test data:
    test[columns] = normalize(test[columns])
    # merge the training and test data:
    dataframe: pd.DataFrame = pd.concat([train, test], axis=0)

    return dataframe


@measure_time
def scaling_main() -> None:
    """
    Main method to execute the scaling library
    @param: filename: str: default = project_2_dataset_unsupervised_imputation.csv: the name of the csv file to scale.
    :return: None: execute the scaling library.
    """
    # get all the csv files in the missing_values_handled folder:
    csv_files: list[Path] = list(missing_values_path.glob("*.csv"))

    # iterate over the csv files:
    for file in csv_files:
        # read the csv file:
        dataframe: pd.DataFrame = pd.read_csv(file)

        # get the missing values method used to impute the data:
        missing_values_method = file.stem.split("_")[3:]

        methods = ["Standard", "MinMax", "Robust"]

        for method in methods:

            if method in ["Standard", "Robust"]:
                # in this case we need to transform the data to be approximately normally distributed:
                dataframe = skewness_main(file, suppress_print=True)

            x_train, x_test, y_train, y_test = split_data(dataframe, "default", validation=False)

            dataframe: pd.DataFrame = scale_data(x_train, x_test, x_train.columns, method=method)

            # add back the target column:
            dataframe["default"] = pd.concat([y_train, y_test], axis=0)

            # Save the scaled data in the scaled_datasets folder:
            dataframe.to_csv(Path(scaled_datasets_path,
                                  f"project_2_dataset_{method}_scaling_{'_'.join(missing_values_method)}.csv"),
                             index=False)

            # Normalize the data if the method is Standard or Robust, since min-max scaling is already normalized:
            if method in ["Standard", "Robust"]:
                dataframe: pd.DataFrame = normalize_data(x_train, x_test, x_train.columns)

                # add back the target column:
                dataframe["default"] = pd.concat([y_train, y_test], axis=0)

                # Save the normalized data:
                dataframe.to_csv(Path(scaled_datasets_path,
                                      f"project_2_dataset_normalized_{method}_{'_'.join(missing_values_method)}.csv"),
                                 index=False)


# Driver code:
if __name__ == "__main__":
    scaling_main()
