# Auxiliary library to perform transformations on the data to reduce skewness
from pathlib import Path

# Libraries:
import numpy as np
import pandas as pd
from scipy.stats import boxcox


def boxcox_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Boxcox transform the columns in the dataframe, following the SciPy implementation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    :param df: the dataframe to be analyzed
    :param columns: the columns to be analyzed
    :return: pd.DataFrame: the boxcox transformed columns specified.
    """
    df = df.copy()
    df[columns] = df[columns].apply(lambda x: pd.Series(boxcox(x + 1)[0]))
    return df


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Log transform the columns in the dataframe
    :param df: the dataframe to be analyzed
    :param columns: List['str']: the columns to be analyzed
    :return: pd.DataFrame: the log transformed columns specified.
    """
    df = df.copy()
    df[columns] = df[columns].apply(lambda x: np.log(x + 1))
    return df


def sqrt_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Square root transform the columns in the dataframe, following the Numpy implementation:
    https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
    :param df:  the dataframe to be analyzed
    :param columns:  the columns to be analyzed
    :return:  the square root transformed columns specified.
    """
    df = df.copy()
    df[columns] = df[columns].apply(lambda x: np.sqrt(x))
    return df


def cube_root_transform(df, columns):
    """
    Cube root transform the columns in the dataframe, following the Numpy implementation:
    https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html
    :param df: the dataframe to be analyzed
    :param columns: the columns to be analyzed
    :return: the cube root transformed columns specified.
    """
    df = df.copy()
    df[columns] = df[columns].apply(lambda x: np.cbrt(x))
    return df


def estimate_skewness(df, columns):
    """
    Estimate the skewness of the columns in the dataframe, using Pandas implementation:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.skew.html
    :param df: the dataframe to be analyzed
    :param columns: the columns to be analyzed
    :return: the skewness of the columns specified.
    """
    df = df.copy()
    return df[columns].skew()


def detect_outliers(dataframe: pd.DataFrame) -> None:
    """
    Detect outliers in the dataset.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    :return: None, prints the outliers found with each method.
    """
    # other outlier detection for numerical features:

    # Inter-quartile range method:
    print("Inter-quartile range method, number of outliers detected for each feature:")
    for feature in dataframe.columns:
        if dataframe[feature].dtypes != 'category' and feature != "id":
            # calculate the first and third quartile:
            q1 = dataframe[feature].quantile(0.25)
            q3 = dataframe[feature].quantile(0.75)
            # calculate the interquartile range:
            iqr = q3 - q1
            # calculate the lower and upper bound:
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            # print the number of outliers:
            print(f"{feature}: "
                  f"{len(dataframe[(dataframe[feature] < lower_bound) | (dataframe[feature] > upper_bound)])}")

    # Z-score method:
    print("--------------------")
    print("Z-score method, number of outliers detected for each feature:")
    for feature in dataframe.columns:
        if dataframe[feature].dtypes != 'category' and feature != "id":
            # calculate the mean:
            mean = dataframe[feature].mean()
            # calculate the standard deviation:
            std = dataframe[feature].std()
            # calculate the z-score:
            z_score = (dataframe[feature] - mean) / std
            # print the number of outliers:
            print(f"{feature}: {len(dataframe[(z_score > 3) | (z_score < -3)])}")

    # we have a lot of outliers and skewness in the numerical features.
    # TODO: Davide, Fabio, solutions? Maybe log transformation?
    #  POST: I added a skewness library with methods to detect and mitigate skewness.



