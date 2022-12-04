# Auxiliary library to perform transformations on the data to reduce skewness
from pathlib import Path

# Libraries:
import numpy as np
import pandas as pd
from auxiliary.method_timer import measure_time
from sklearn.preprocessing import PowerTransformer

# Global variables:
from config import logs_path
from modelling.train_test_validation_split import split_data

if not logs_path.exists():
    logs_path.mkdir(parents=True)


# Functions:
def power_transform(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame, columns: list[str],
                    method: str) -> pd.DataFrame:
    """
    Power transform the columns in the dataframe, following the Scikit-learn implementation:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    @param df: the dataframe to be analyzed
    @param columns: the columns to be analyzed
    @param method: the method to be used, either "yeo-johnson" or "box-cox"
    :return: the power transformed columns specified.
    """
    train = train.copy()
    validate = validate.copy()
    test = test.copy()
    # create the transformer:
    pt = PowerTransformer(method=method)
    # fit the transformer:
    pt.fit(train[columns])
    # transform the columns:
    train[columns] = pt.transform(train[columns])
    validate[columns] = pt.transform(validate[columns])
    test[columns] = pt.transform(test[columns])
    # merge the dataframes:
    df = pd.concat([train, validate, test])
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
    :param columns: the columns to be analyzed, all if "all"
    :return: the skewness of the columns specified.
    """
    df = df.copy()
    if columns == "all":
        # return every column other than the id, reverse the order to have the most skewed first:
        return df.skew(numeric_only=True).sort_values(ascending=False)
    return df[columns].skew(numeric_only=True)


def detect_outliers_statistically(dataframe: pd.DataFrame) -> None:
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


def handle_outliers_statistically(dataframe: pd.DataFrame, method: str, strategy: str) -> pd.DataFrame:
    """
    Drop outliers in the dataset.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    @param method: str: the method to be used to handle outliers, supported methods are:
        - "iqr": Inter-quartile range method.
        - "z_score": Z-score method with 3 sigma as the normal range.
    @param strategy: str: the strategy to be used to handle outliers, supported strategies are:
        - "drop": drop the outliers.
        - "replace": replace the outliers with the median, in this case it's better than the mean due to skewness.
        - "cap": cap the outliers with the upper and lower bound.
        - "keep": keep the outliers.
    :return: pd.DataFrame: the dataframe without outliers.
    """
    # Inter-quartile range method:
    if method == "iqr":
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
                # drop the outliers:
                if strategy == "drop":
                    dataframe = dataframe[(dataframe[feature] > lower_bound) & (dataframe[feature] < upper_bound)]
                # replace the outliers with the median:
                elif strategy == "replace":
                    dataframe[feature] = dataframe[feature].apply(lambda x: dataframe[feature].median()
                    if x < lower_bound or x > upper_bound else x)
                # cap the outliers with the upper and lower bound:
                elif strategy == "cap":
                    dataframe[feature] = dataframe[feature].apply(lambda x: upper_bound if x > upper_bound
                    else lower_bound if x < lower_bound else x)
                # keep the outliers:
                elif strategy == "retain":
                    pass
                # error handling:
                else:
                    raise ValueError("The strategy is not supported.")

    # Z-score method:
    elif method == "z-score":
        for feature in dataframe.columns:
            if dataframe[feature].dtypes != 'category' and feature != "id":
                # calculate the mean:
                mean = dataframe[feature].mean()
                # calculate the standard deviation:
                std = dataframe[feature].std()
                # calculate the z-score:
                z_score = (dataframe[feature] - mean) / std
                if strategy == "drop":
                    # drop the outliers:
                    dataframe = dataframe.drop(dataframe[(z_score > 3) | (z_score < -3)].index)
                elif strategy == "replace":
                    # replace the outliers with the median:
                    dataframe[feature] = dataframe[feature].apply(lambda x: dataframe[feature].median()
                    if x < mean - 3 * std or x > mean + 3 * std else x)
                elif strategy == "cap":
                    # cap the outliers with the upper and lower bounds:
                    dataframe.loc[(z_score > 3) | (z_score < -3), feature] = \
                        dataframe.loc[(z_score > 3) | (z_score < -3), feature].apply(
                            lambda x: lower_bound if x < lower_bound else upper_bound)
                # keep the outliers:
                elif strategy == "retain":
                    pass
                # error handling:
                else:
                    raise ValueError("The strategy is not supported.")

    return dataframe


# The main function, for now only on the features.
def shift_negative_values(features: pd.DataFrame) -> pd.DataFrame:
    """
    Shifts the negative values of the features to the positive values.
    @param features: the features dataframe to be analyzed
    :return: The features dataframe where the negative values have been shifted to the positive values.
    """
    # We find the minimum value of each feature
    for feature in features.columns:
        # exclude the categorical features and the id:
        if features[feature].dtype != "category":
            # shift the negative values to the positive values:
            features[feature] = features[feature].apply(lambda x: x - features[feature].min() + 1 if x < 0 else x)
    return features


@measure_time
def skewness_main(filename: str, suppress_print=False) -> pd.DataFrame:
    """
    The main function of the skewness module.
    @param filename: str: the name of the file for which the skewness will be handled.
    @param suppress_print: bool: if True, the print statements will be suppressed.
    @return: pd.DataFrame: the dataframe with the skewness handled.
    """
    df = pd.read_csv(filename)

    # assign the categorical features:
    categorical_features: list[str] = ["gender", "education", "marriage", "pay_stat_sep", "pay_stat_aug",
                                       "pay_stat_jul",
                                       "pay_stat_jun", "pay_stat_may", "pay_stat_apr", "default"]

    df = df.astype({feature: "category" for feature in categorical_features})

    # set the id as the index if it exists:
    if "id" in df.columns:
        df.set_index("id", inplace=True)

    # estimate the skewness of the features:
    if not suppress_print:
        print(estimate_skewness(df, "all"))

    # get the numerical features names by excluding the categorical features and the id:
    numerical_features: list[str] = df.select_dtypes(exclude="category").columns

    # apply the yeo-johnson transformation, we selected it as the best transformation since it handles negative values
    # and it's the most robust transformation.
    # See I.K. Yeo and R.A. Johnson, “A new family of power transformations to improve normality or symmetry.”
    # Biometrika, 87(4), pp.954-959, (2000).
    # Split the data in training, validation and test sets, no data leakage:
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df, "default", validation=True)
    training = pd.concat([x_train, y_train], axis=1)
    validation = pd.concat([x_val, y_val], axis=1)
    testing = pd.concat([x_test, y_test], axis=1)

    # apply the yeo-johnson transformation:
    features = power_transform(training, validation, testing, numerical_features, "yeo-johnson")

    if not suppress_print:
        print(estimate_skewness(df, "all"))

    # substitute the numerical features with the transformed features:
    df[numerical_features] = features[numerical_features]

    return df

