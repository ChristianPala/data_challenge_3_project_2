# Library to preprocess the creditor data from project 2 of the Data Challenge III course.
# Libraries:
# Data manipulation:
from pathlib import Path
import shutil
from typing import List

import numpy as np
import pandas as pd
# Timing
from auxiliary.method_timer import measure_time
# Imputation:
from preprocessing.missing_values_handling import handle_missing_values
# Modelling
from modelling.train_test_validation_split import split_data

# Console options:
# pd.set_option('display.max_columns', None)

# Global variables:
from config import excel_file, missing_values_path

if not excel_file.is_file():
    raise FileNotFoundError('The dataset is not in the directory. Please download it from the course website.')


# Functions:
def load_data() -> pd.DataFrame:
    """
    Load the dataset from the Excel file.
    :return: pd.DataFrame: the dataframe containing the dataset.
    """
    # Load the dataset, ignore the first row:
    dataframe: pd.DataFrame = pd.read_excel(excel_file, sheet_name='Data', skiprows=1)

    return dataframe


def inspect(dataframe: pd.DataFrame) -> None:
    """
    Perform an initial inspection of the dataset.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    :return: None
    """
    # Print the first 5 rows:
    print(dataframe.head())

    # Print the last 5 rows:
    print(dataframe.tail())

    # Print the shape of the dataset:
    print(dataframe.shape)

    # Print the columns of the dataset:
    print(dataframe.columns)

    # Print the data types of the dataset:
    print(dataframe.dtypes)

    # Print the info of the dataset:
    print(dataframe.info())

    # Print the summary statistics of the dataset:
    print(dataframe.describe())


def rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Rename selected columns of the dataframe.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    :return: pd.DataFrame: the dataframe with the renamed columns.
    """

    # map all names to lower case:
    dataframe.columns = dataframe.columns.str.lower()

    # improve readability:
    dataframe.rename(columns={"pay_0": "pay_stat_sep", "pay_2": "pay_stat_aug", "pay_3": "pay_stat_jul",
                              "pay_4": "pay_stat_jun", "pay_5": "pay_stat_may", "pay_6": "pay_stat_apr",
                              "bill_amt1": "bill_amt_sep", "bill_amt2": "bill_amt_aug", "bill_amt3": "bill_amt_jul",
                              "bill_amt4": "bill_amt_jun", "bill_amt5": "bill_amt_may", "bill_amt6": "bill_amt_apr",
                              "pay_amt1": "pay_amt_sep", "pay_amt2": "pay_amt_aug", "pay_amt3": "pay_amt_jul",
                              "pay_amt4": "pay_amt_jun", "pay_amt5": "pay_amt_may", "pay_amt6": "pay_amt_apr",
                              "sex": "gender", "default payment next month": "default"}, inplace=True)

    return dataframe


def remap_education_and_marriage(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Correct the values of the education column.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    :return: pd.DataFrame: the dataframe with the categorical features as categories.
    """
    # Education has 0, 5, 6 as values, which are not in the description of the dataset, after clarification
    # with professor Mitrovic we will map them 5 and 6 to 4 (others), 0 is a missing value:
    # which is the value for "others":
    dataframe["education"] = dataframe["education"].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4})
    # Marriage has 0 as a value, which is not in the description of the dataset, after clarification
    # we consider it as a missing value and solve the issue before assigning the categories.
    # Properly assign the 14 education and 54 marriage missing values to NaN values:
    dataframe["education"].replace(0, np.nan, inplace=True)
    dataframe["marriage"].replace(0, np.nan, inplace=True)

    return dataframe


@measure_time
def preprocessor_main(suppress_print=False, missing_values_dominant_strategies: List[str] = None) -> None:
    """
    Main function. Load the dataset, preprocess it and save it.
    @param suppress_print: bool: if True, suppress the prints to the console.
    @param missing_values_dominant_strategies: List[str]: the dominant strategies to use for the missing values,
    if they exists. If None, all the strategies will be used.
    :return: Save the preprocessed dataset as a csv and pickle file.
    """
    # Load the dataset:
    dataframe: pd.DataFrame = load_data()

    # Cleaning paths:
    if missing_values_path.exists() and missing_values_path.is_dir():
        shutil.rmtree(missing_values_path)
    missing_values_path.mkdir(parents=True, exist_ok=True)

    # inspect the dataset:
    if not suppress_print:
        print("Dataset information:")
        print("-" * 100)
        inspect(dataframe)
        print("-" * 100)

        # check for missing values and duplicates:
        # Check for missing values:

        missing_values: int = dataframe.isnull().sum().sum() + len(dataframe[dataframe['EDUCATION'] == 0]) + \
                              len(dataframe[dataframe['MARRIAGE'] == 0])

        print(f"Missing values: {missing_values} ")
        print("-" * 100)
        # Check for duplicated rows:
        print(f"Duplicated rows: {dataframe.duplicated().sum()}")
        print("-" * 100)
        """
    No explicit missing values, but there are s few values in the dataset, which are not in the description, professor
    Mitrovic clarified that if their values is 0, are missing values, so we will handle them as such.
    """
    # Rename the columns:
    dataframe = rename_columns(dataframe)

    # Assign categories to the categorical features:
    dataframe = remap_education_and_marriage(dataframe).copy()

    # split the dataset into training, testing and validation:
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(dataframe, 'default', validation=True)

    # combine training, validation and testing:
    training_dataframe = pd.DataFrame(pd.concat([x_train, y_train], axis=1))
    validation_dataframe = pd.DataFrame(pd.concat([x_val, y_val], axis=1))
    testing_dataframe = pd.DataFrame(pd.concat([x_test, y_test], axis=1))


    if not missing_values_dominant_strategies:
        # missing values imputation methods:
        methods: list[str] = ["drop", "most_frequent_imputation", "supervised_imputation", "unsupervised_imputation"]

        # preprocess the training, validation and testing datasets:
        for method in methods:
            # preprocess the training dataset:
            training_dataframe = handle_missing_values(training_dataframe, method)
            # preprocess the validation dataset:
            validation_dataframe = handle_missing_values(validation_dataframe, method)
            # preprocess the testing dataset:
            testing_dataframe = handle_missing_values(testing_dataframe, method)

            # merge the training, validation and testing datasets:
            dataframe = pd.DataFrame(pd.concat([training_dataframe, validation_dataframe, testing_dataframe],
                                               axis=0))
            # save the preprocessed dataset in the missing_values_path:
            dataframe.to_csv(Path(missing_values_path, f"project_2_dataset_{method}.csv"), index=False)

            if not suppress_print:
                print("Preprocessing completed.")
                print("-" * 100)

    else:
        for strategy in missing_values_dominant_strategies:
            # preprocess the training dataset:
            training_dataframe = handle_missing_values(training_dataframe, strategy)
            # preprocess the validation dataset:
            validation_dataframe = handle_missing_values(validation_dataframe, strategy)
            # preprocess the testing dataset:
            testing_dataframe = handle_missing_values(testing_dataframe, strategy)

            # merge the training, validation and testing datasets:
            dataframe = pd.DataFrame(pd.concat([training_dataframe, validation_dataframe, testing_dataframe],
                                               axis=0))
            # save the preprocessed dataset in the missing_values_path:
            dataframe.to_csv(Path(missing_values_path, f"project_2_dataset_{strategy}.csv"),
                             index=False)

            if not suppress_print:
                print("Preprocessing completed.")
                print("-" * 100)


if __name__ == '__main__':
    preprocessor_main()
