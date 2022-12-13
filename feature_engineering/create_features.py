# library to feature engineering some features from the project 2 dataset in the Data Challenge III course,
# SUPSI Bachelor in Data Science, 2022.

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
data_path = Path("..", "data")
missing_values_handled_path = Path(data_path, "missing_values_handled")


# Functions:
def pay_status_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column with the cumulative sum of the pay status columns.
    @param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.copy()
    df['pay_status_total'] = df[['pay_stat_sep', 'pay_stat_aug', 'pay_stat_jul', 'pay_stat_jun', 'pay_stat_may',
                                 'pay_stat_apr']].sum(axis=1)

    return df


def total_bill_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column with the total amount of the bill.
    @param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.copy()
    df['total_bill_amount'] = df[['bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun', 'bill_amt_may',
                                  'bill_amt_apr']] \
        .sum(axis=1)

    return df


def total_paid_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a new column with the total amount paid.
    @param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = df.copy()
    df['total_paid_amount'] = df[['pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun', 'pay_amt_may',
                                  'pay_amt_apr']].sum(axis=1)

    return df


@measure_time
def feature_engineering_main(overwrite_original: bool = False) -> None:
    """
    This function creates the new features and saves the dataset with the new features on all the csv files in the data
    folder.
    @param overwrite_original: bool: if True, the original csv files are overwritten with the new features.
    :return: None, saves the new features in the csv files.
    """

    # get all the csv files in the data folder:
    csv_files = list(missing_values_handled_path.glob('*.csv'))

    # loop over the csv files:
    for csv_file in csv_files:
        # read the csv file:
        df = pd.read_csv(csv_file)

        # create the new features:
        df = pay_status_cumulative(df)
        df = total_bill_amount(df)
        df = total_paid_amount(df)

        # save the new features:
        file_name = csv_file.stem + "_augmented.csv"
        df.to_csv(Path(missing_values_handled_path, file_name), index=False)

        # delete the original csv file:
        if overwrite_original:
            csv_file.unlink()


# Driver:
if __name__ == '__main__':
    feature_engineering_main()
