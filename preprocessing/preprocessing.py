# File to preprocess the creditor data from project 2 of the Data Challenge III course.

# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# Data loading:
from data_loading import load_and_save_data


if __name__ == '__main__':
    # Load the data:
    df: pd.DataFrame = load_and_save_data()

    # initial data view:
    # visualize all the columns in the console:
    pd.set_option('display.max_columns', None)

    # print some info about the dataset:
    print(df.head())
    print(df.info())
    print(df.describe())

    # check missing values:
    print(df.isnull().sum())

    # check the data types:
    print(df.dtypes)

    # initial name consistency:
    # rename the column PAY_0 to PAY_1:
    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)

    # categorical variables:
    categorical_features: list[str] = ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                                       "PAY_6"]

    numerical_features: list[str] = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                                     "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
                                     "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    target_feature: str = "default payment next month"

    # check the cardinality of the categorical variables:
    for feature in categorical_features:
        print(f"{feature}: {df[feature].unique()}")

    # Save the data:
    df.to_csv(Path('..', 'data', 'Project_2_Dataset.csv'), index=False)

