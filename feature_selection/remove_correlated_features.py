# Library to do simple feature selection, by leaving only cumulative features for
# the highest correlated features:

# Libraries
import pandas as pd
from pathlib import Path


def simplify_dataset(train_path: Path, validation_path: Path) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    This function simplifies the dataset by removing the columns with the highest correlation.
    :return: None
    """
    # Load the data:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path)

    x_train = train_df.drop(columns=['default'])
    y_train = train_df['default']

    x_val = val_df.drop(columns=['default'])
    y_val = val_df['default']

    # since we already created total pay_status_cumulative, we can remove the other pay_status columns:
    x_train = x_train.drop(columns=['pay_stat_sep', 'pay_stat_aug', 'pay_stat_jul', 'pay_stat_jun',
                                    'pay_stat_may', 'pay_stat_apr'])
    x_val = x_val.drop(columns=['pay_stat_sep', 'pay_stat_aug', 'pay_stat_jul', 'pay_stat_jun',
                                'pay_stat_may', 'pay_stat_apr'])

    # since we already created total_bill_amount, we can remove the other bill_amt columns:
    x_train = x_train.drop(columns=['bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun',
                                    'bill_amt_may', 'bill_amt_apr'])

    x_val = x_val.drop(columns=['bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun',
                                'bill_amt_may', 'bill_amt_apr'])

    # since we already created total_paid_amount, we can remove the other pay_amt columns:
    x_train = x_train.drop(columns=['pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun',
                                    'pay_amt_may', 'pay_amt_apr'])

    x_val = x_val.drop(columns=['pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun',
                                'pay_amt_may', 'pay_amt_apr'])

    return x_train, y_train, x_val, y_val

# We did not expand on feature selection, since all features are not strongly correleted with the target,
# it does not make a lot of sense to start removing features when we only have 23 features to begin with.
# The exception is for the simple models, where we can use the cumulative features instead of the individual features
# to simplify the task..