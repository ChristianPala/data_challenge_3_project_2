# Auxiliary library to perform transformations on the data to reduce skewness
from pathlib import Path

# Libraries:
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


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
    :param columns: the columns to be analyzed, all if "all"
    :return: the skewness of the columns specified.
    """
    df = df.copy()
    if columns == "all":
        # return every column other than the id, reverse the order to have the most skewed first:
        return df.drop(columns=["id"]).skew(numeric_only=True).sort_values(ascending=False)
    return df[columns].skew(numeric_only=True)


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


def handle_outliers(dataframe: pd.DataFrame, method: str, strategy: str) -> pd.DataFrame:
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


def smote_transform(train: pd.DataFrame, target: str) -> tuple[np.array, np.array]:
    """
    Transformation using the imbalance learn implementation of SMOTE(synthetic minority over-sampling technique):
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    The original paper on SMOTE suggested combining SMOTE with random under-sampling of the majority class.:
    https://arxiv.org/abs/1106.1813
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: The target and the features after the SMOTE transformation
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    X = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our over-sampler and under-sampler
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)

    # Using a pipeline to streamline the process
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # Transform the dataset
    x, y = pipeline.fit_resample(X, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    return x, y


def undersample_transform(train: pd.DataFrame, target: str) -> tuple[np.array, np.array]:
    """
    Transformation using the imbalance learn implementation of Random under-sampling:
    https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: The target and the features dataframes where under-sampling has been applied.
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    x = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our under-sampler
    under = RandomUnderSampler(sampling_strategy=0.5)

    # Transform the dataset
    x, y = under.fit_resample(x, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    return x, y


def oversample_transform(train: pd.DataFrame, target: str) -> tuple[np.array, np.array]:
    """
    Transformation using the imbalance learn implementation of Random over-sampling:
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: The target and the features dataframes where over-sampling has been applied.
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    x = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our over-sampler
    over = RandomOverSampler(sampling_strategy=0.1)

    # Transform the dataset
    x, y = over.fit_resample(x, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    return x, y


# The main function, for now only on the features.
def main() -> None:
    df = pd.read_pickle(Path('..', 'data', 'project_2_dataset.pkl'))
    # print the skewness of the dataset:
    print("Skewness of the dataset:")
    print(estimate_skewness(df, "all"))
    print("--------------------")

    # print the outliers found with each method:
    detect_outliers(df)
    print("--------------------")
    # apply box-cox transformation to the non-negative numerical features of the dataset:
    numerical = df.select_dtypes(exclude='category').columns
    # drop the id column:
    numerical = numerical.drop("id")
    # keep only the non-negative numerical features:
    positive_numerical = numerical[df[numerical].min() >= 0]
    negative_numerical = numerical[df[numerical].min() < 0]
    # apply box-cox transformation:
    df = boxcox_transform(df, positive_numerical)
    # shift the negative numerical features by the minimum value + 1:
    df[negative_numerical] = df[negative_numerical].apply(lambda x: x - x.min() + 1)
    # apply box-cox transformation again:
    df = boxcox_transform(df, negative_numerical)
    # print the skewness of the dataset after the transformation:
    print("Skewness of the dataset after the transformation:")
    print(estimate_skewness(df, "all"))
    print("--------------------")
    # print the outliers found with each method:
    detect_outliers(df)
    print("--------------------")


if __name__ == '__main__':
    main()
