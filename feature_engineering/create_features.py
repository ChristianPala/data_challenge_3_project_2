# library to feature engineering some features from the project 2 dataset in the Data Challenge III course,
# SUPSI Bachelor in Data Science, 2022.

# Libraries:
import pandas as pd
import numpy as np

# Data manipulation:
from pathlib import Path

from auxiliary.method_timer import measure_time

# global variables:
data_path = Path("..", "data")


# Functions:
def pay_status_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column with the cumulative sum of the pay status columns.
    @param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.copy()
    df['pay_status_total'] = df[['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].sum(axis=1)

    return df


def total_bill_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column with the total amount of the bill.
    @param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.copy()
    df['total_bill_amount'] = df[['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']] \
        .sum(axis=1)

    return df


def total_paid_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column with the total amount paid.
    @param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.copy()
    df['total_paid_amount'] = df[['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']].sum(axis=1)

    return df


@measure_time
def main() -> None:
    """
    This function creates the new features and saves the dataset with the new features on all the csv files in the data
    folder.
    :return: None, saves the new features in the csv files.
    """

    # get all the csv files in the data folder:
    csv_files = list(data_path.glob('*.csv'))

    # loop over the csv files:
    for csv_file in csv_files:
        # read the csv file:
        df = pd.read_csv(csv_file)

        # create the new features:
        df = pay_status_cumulative(df)
        df = total_bill_amount(df)
        df = total_paid_amount(df)

        # save the new dataset:
        df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    main()
#  Todo: Davide, Fabio, ideas for other new features?
