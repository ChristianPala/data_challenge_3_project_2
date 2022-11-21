# Library to preprocess the creditor data from project 2 of the Data Challenge III course.

"""
Information about the dataset from professor Mitrovic in the Data Challenge III course (Project 2)
introduction slides:

Current available amount of credit in NT dollars:
•LIMIT_BAL: Amount of the given credit: it includes both the individual

Demographic:
consumer credit and his/her family (supplementary) credit
•GENDER: Gender (1 = male; 2 = female).
•EDUCATION: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
•MARRIAGE: Marital status (1 = married; 2 = single; 3 = others)
•AGE: Age (year)
* Further clarification, 0 can be interpreted as a missing value.

•History of past payment:
•PAY_1 - PAY_6: Past monthly payment records (from April to September, 2005)
as follows: PAY_1 = the repayment status in September, 2005;
PAY_2 = the repayment status in August, 2005; . . .;
PAY_6 = the repayment status in April, 2005.
•The measurement scale for the repayment status is:
 -2: No consumption; -1: Paid in full; 0: The use of revolving credit;
 1 = payment delay for one month; 2 = payment delay for two months; . . .;
 8 = payment delay for eight months; 9 = payment delay for nine months and above

•Amount of bill statement:
•BILL_AMT1 - BILL_AMT6: BILL_AMT1 = amount of bill statement in September, 2005;
BILL_AMT2 = amount of bill statement in August, 2005; . . .;
BILL_AMT6 = amount of bill statement in April, 2005.•Amount of previous payment:
•PAY_AMT1 - PAY_AMT6: PAY_AMT1 = amount paid in September, 2005;
PAY_AMT2 = amount paid in August, 2005; . . .; PAY_AMT6 = amount paid in April, 2005
"""

# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# imputation:
from missing_values_handling import handle_missing_values
from auxiliary.method_timer import measure_time

# Console options:
# pd.set_option('display.max_columns', None)


# Global variables:
# Path to the dataset:
data_path: Path = Path('..', 'data')
excel_file: Path = Path(data_path, 'Project 2 Dataset.xls')


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

    # TODO: Davide, Fabio, this is the logical order I used, should we compact it?
    # Rename the columns for consistency and appropriateness:
    dataframe.rename(columns={"PAY_0": "PAY_1", "SEX": "GENDER"}, inplace=True)

    # rename the target feature:
    dataframe.rename(columns={"default payment next month": "DEFAULT"}, inplace=True)

    # map all names to lower case:
    dataframe.columns = dataframe.columns.str.lower()

    # improve readability:
    dataframe.rename(columns={"pay_1": "pay_stat_sep", "pay_2": "pay_stat_aug", "pay_3": "pay_stat_jul",
                              "pay_4": "pay_stat_jun", "pay_5": "pay_stat_may", "pay_6": "pay_stat_apr",
                              "bill_amt1": "bill_amt_sep", "bill_amt2": "bill_amt_aug", "bill_amt3": "bill_amt_jul",
                              "bill_amt4": "bill_amt_jun", "bill_amt5": "bill_amt_may", "bill_amt6": "bill_amt_apr",
                              "pay_amt1": "pay_amt_sep", "pay_amt2": "pay_amt_aug", "pay_amt3": "pay_amt_jul",
                              "pay_amt4": "pay_amt_jun", "pay_amt5": "pay_amt_may", "pay_amt6": "pay_amt_apr"},
                     inplace=True)

    return dataframe


def assign_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Assign categories to the categorical features.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    :return: pd.DataFrame: the dataframe with the categorical features as categories.
    """
    # Name of the categorical features from the dataset information:
    categorical_features: list[str] = ["gender", "education", "marriage", "pay_stat_sep", "pay_stat_aug",
                                       "pay_stat_jul",
                                       "pay_stat_jun", "pay_stat_may", "pay_stat_apr", "default"]

    # Education has 0, 5, 6 as values, which are not in the description of the dataset, after clarification
    # with professor Mitrovic we will map them 5 and 6 to 4 (others), 0 is a missing value:
    # which is the value for "others":
    dataframe["education"] = dataframe["education"].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4})
    # Marriage has 0 as a value, which is not in the description of the dataset, after clarification
    # we consider it as a missing value and solve the issue before assigning the categories.

    # Assign categories to the categorical features:
    dataframe[categorical_features] = dataframe[categorical_features].astype('category')

    return dataframe

@measure_time
def main() -> None:
    """
    Main function. Load the dataset, preprocess it and save it.
    :return: Save the preprocessed dataset as a csv and pickle file.
    """
    # Load the dataset:
    dataframe: pd.DataFrame = load_data()

    # inspect the dataset:
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

    # Handle missing values, save a copy for all methods supported:
    methods: list[str] = ["drop", "ignore", "most_frequent_imputation",
                          "supervised_imputation", "unsupervised_imputation"]
    for method in methods:
        # make a copy of the dataframe:
        dataframe_copy: pd.DataFrame = dataframe.copy()
        # handle missing values:
        dataframe_copy = handle_missing_values(dataframe_copy, method)
        # Assign categories to the categorical features:
        dataframe_copy = assign_categories(dataframe_copy)

        # Save the data to pickle to maintain the categories:
        dataframe_copy.to_pickle(Path(data_path, f"project_2_dataset_{method}.pkl"))

        # Save the data to csv:
        dataframe_copy.to_csv(Path(data_path, f"project_2_dataset_{method}.csv"), index=False)

    print("Done!")
    print("-" * 100)


if __name__ == '__main__':
    main()
