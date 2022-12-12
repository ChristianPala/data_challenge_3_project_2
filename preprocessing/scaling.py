# Library to scale and normalize the data.

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
import shutil
# Scaling and normalization:
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, normalize
# Modelling:
from modelling.train_test_validation_split import split_data
from preprocessing.features_skeweness_handling import skewness_main
# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
from config import scaled_datasets_path, missing_values_path


def scale_data(training: pd.DataFrame, validation: pd.DataFrame, testing: pd.DataFrame,
               columns: list[str], method: str = "Standard") -> pd.DataFrame:
    """
    Scale the data using the specified method, taken from Scikit-learn's preprocessing module:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    @param training: pd.DataFrame: the dataframe containing the training data.
    @param validation: pd.DataFrame: the dataframe containing the validation data.
    @param testing: pd.DataFrame: the dataframe containing the test data.
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
    scaler.fit(training[columns])
    # scale the training data:
    training[columns] = scaler.transform(training[columns])

    # scale the validation data:
    validation[columns] = scaler.transform(validation[columns])

    # scale the test data:
    testing[columns] = scaler.transform(testing[columns])

    # merge the training and test data:
    dataframe: pd.DataFrame = pd.concat([training, validation, testing], axis=0)

    return dataframe


def normalize_data(training: pd.DataFrame, validation: pd.DataFrame, testing: pd.DataFrame,
                   columns: list[str]) -> (pd.DataFrame, pd.DataFrame):
    """
    Normalize the data using Scikit-learn's normalize method:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    @param training: the dataframe containing the training data.
    @param validation: the dataframe containing the validation data.
    @param testing: the dataframe containing the test data.
    @param columns: the list of columns to normalize.
    @return: the normalized dataset
    """
    # normalize the training data:
    training[columns] = normalize(training[columns])
    # normalize the validation data:
    validation[columns] = normalize(validation[columns])
    # normalize the test data:
    testing[columns] = normalize(testing[columns])
    # merge the training and test data:
    dataframe: pd.DataFrame = pd.concat([training, validation, testing], axis=0)

    return dataframe


@measure_time
def scaling_main(dominant_scaling_strategies: [str] = None, save_non_normalized: bool = False) -> None:
    """
    Main method to execute the scaling library
    @param dominant_scaling_strategies: the dominant scaling strategy to use, if it exists. If it does not exist,
    all scaling strategies will be used.
    @param save_non_normalized: whether to also save the non-normalized datasets.
    :return: None: execute the scaling library.
    """

    if not missing_values_path.exists():
        raise FileNotFoundError("The missing values path does not exist. Cannot scale the data.")

    # get all the csv files in the missing_values_handled folder:
    csv_files: list[Path] = list(missing_values_path.glob("*.csv"))

    # Cleaning paths:
    if scaled_datasets_path.exists() and scaled_datasets_path.is_dir():
        shutil.rmtree(scaled_datasets_path)
    scaled_datasets_path.mkdir(parents=True, exist_ok=True)

    # iterate over the csv files:
    for file in csv_files:
        # read the csv file:
        dataframe: pd.DataFrame = pd.read_csv(file)

        # get the missing values method used to impute the data:
        missing_values_method = file.stem.split("_")[3:]

        methods = ["standard_scaler", "minmax_scaler", "robust_scaler"]

        if not dominant_scaling_strategies:

            for method in methods:

                if method in ["standard_scaler", "robust_scaler"]:
                    # in this case we need to transform the data to be approximately normally distributed:
                    dataframe = skewness_main(file, suppress_print=True)

                x_train, x_val, x_test, y_train, y_val, y_test = split_data(dataframe, "default", validation=True)

                dataframe: pd.DataFrame = scale_data(x_train, x_val, x_test, x_train.columns, method=method)

                # add back the target column:
                dataframe["default"] = pd.concat([y_train, y_val, y_test], axis=0)

                # Save the scaled data in the scaled_datasets folder if also the non-normalized data is saved:
                if save_non_normalized:
                    dataframe.to_csv(Path(scaled_datasets_path,
                                          f"project_2_dataset_{method}_scaling_{'_'.join(missing_values_method)}.csv"),
                                     index=False)

                # Normalize the data if the method is Standard or Robust, since min-max scaling is already normalized:
                if method in ["standard_scaler", "robust_scaler"]:
                    dataframe: pd.DataFrame = normalize_data(x_train, x_val, x_test, x_train.columns)

                    # add back the target column:
                    dataframe["default"] = pd.concat([y_train, y_val, y_test], axis=0)

                    # Save the normalized data:
                    dataframe.to_csv(Path(scaled_datasets_path,
                                          f"project_2_dataset_normalized_{method}_{'_'.join(missing_values_method)}.csv"),
                                     index=False)
        elif dominant_scaling_strategies:

            for method in dominant_scaling_strategies:

                if method in ["standard_scaler", "robust_scaler"]:
                    # in this case we need to transform the data to be approximately normally distributed:
                    dataframe = skewness_main(file, suppress_print=True)

                x_train, x_val, x_test, y_train, y_val, y_test = split_data(dataframe, "default", validation=True)

                dataframe: pd.DataFrame = scale_data(x_train, x_val, x_test, x_train.columns, method=method)

                # add back the target column:
                dataframe["default"] = pd.concat([y_train, y_val, y_test], axis=0)

                # Save the scaled data in the scaled_datasets folder if also the non-normalized data is saved:
                if save_non_normalized:
                    dataframe.to_csv(Path(scaled_datasets_path,
                                          f"project_2_dataset_{method}_scaling_{'_'.join(missing_values_method)}.csv"),
                                     index=False)

                # Normalize the data if the method is Standard or Robust, since min-max scaling is already normalized:
                if method in ["standard_scaler", "robust_scaler"]:
                    dataframe: pd.DataFrame = normalize_data(x_train, x_val, x_test, x_train.columns)

                    # add back the target column:
                    dataframe["default"] = pd.concat([y_train, y_val, y_test], axis=0)

                    # Save the normalized data:
                    dataframe.to_csv(Path(scaled_datasets_path,
                                          f"project_2_dataset_normalized_{method}_{'_'.join(missing_values_method)}.csv"),
                                     index=False)


# Driver code:
if __name__ == "__main__":
    scaling_main()
