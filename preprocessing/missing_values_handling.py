# Auxiliary library to deal with missing values in the dataset
# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
from pathlib import Path

# Imputing:
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier


def handle_missing_values(dataframe: pd.DataFrame, method: str = "supervised_imputation") -> pd.DataFrame:
    """
    Handle missing values in the dataset, in this instance they are values equal to 0 in the
    education and marriage categorical features..
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    @param method: str: default is supervised_imputation. The method to use to handle the missing values.
    Supports the following strategies:
    - drop,
    - most_frequent_imputation,
    - supervised_imputation
    - unsupervised_imputation.
    :return: pd.DataFrame: the dataframe without missing values.
    """

    # make a copy of the dataframe:
    dataframe: pd.DataFrame = dataframe.copy()

    # check if the dataframe contains missing values, otherwise return the dataframe:
    if dataframe.isnull().sum().sum() == 0 and len(dataframe[dataframe['education'] == 0]) == 0 and \
            len(dataframe[dataframe['marriage'] == 0]) == 0:
        return dataframe

    # store the indexes of the missing values:
    missing_indexes_e = dataframe[dataframe["education"].isna()].index
    missing_indexes_m = dataframe[dataframe["marriage"].isna()].index

    print("Missing values in education: ", len(missing_indexes_e))
    print("Missing values in marriage: ", len(missing_indexes_m))
    if method == "drop":
        # Drop the missing values, i.e. rows with education or marriage equal to 0:
        dataframe.dropna(inplace=True)

    elif method == "most_frequent_imputation":
        # Impute the missing values with the mode:
        dataframe["education"].replace(np.nan, dataframe["education"].mode()[0], inplace=True)
        dataframe["marriage"].replace(np.nan, dataframe["marriage"].mode()[0], inplace=True)

    elif method == "supervised_imputation":
        # Impute the missing values with a supervised method:
        # Using random forest classifier to avoid preprocessing the data:
        """
        Using the random forest classifier from sklearn: 
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html to impute
        the missing values in the education and marriage features.
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        # the test set is the one with the missing values:
        test_set = dataframe[dataframe["education"].isna() | dataframe["marriage"].isna()].copy()
        train_set = dataframe[~(dataframe["education"].isna() | dataframe["marriage"].isna())].copy()
        # separate the features from the target:
        train_set_features = train_set.drop(["education", "marriage"], axis=1)
        train_set_target = train_set[["education", "marriage"]]
        # train the model:
        rf.fit(train_set_features, train_set_target)
        # predict the missing values:
        test_set[["education", "marriage"]] = rf.predict(test_set.drop(["education", "marriage"], axis=1))
        # concatenate the training set and the test set:
        dataframe = pd.concat([train_set, test_set], axis=0)

    elif method == "unsupervised_imputation":
        # Impute the missing values with the most similar instance using KNN:
        """
        using the k-nearest neighbors algorithm from the sklearn library:
        https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html to impute the missing values
        in the education and marriage features.
        """
        knn_imputer = KNNImputer(n_neighbors=1, metric="nan_euclidean")
        # assign NaN to the missing values in education and marriage:
        dataframe["education"].replace(0, np.nan, inplace=True)
        dataframe["marriage"].replace(0, np.nan, inplace=True)
        # impute the missing values:
        dataframe = pd.DataFrame(knn_imputer.fit_transform(dataframe), columns=dataframe.columns)

    else:
        raise ValueError(f"Method {method} not supported, possible choices are 'drop' or 'impute'.")

    if method != "drop":
        # save the change in the original dataframe to a txt file:
        log_path = Path("..", "logs")
        log_path.mkdir(parents=True, exist_ok=True)
        with open(Path(log_path, f"missing_values_{method}.txt"), "w") as file:
            file.write(f"Missing values for education: {missing_indexes_e}\n")
            file.write(f"Missing values for marriage: {missing_indexes_m}\n")
            file.write(f"Missing values imputed with: {method}\n")
            file.write(f"imputed values for education: {str(dataframe['education'][missing_indexes_e].values)}\n")
            file.write(f"imputed values for marriage: {str(dataframe['marriage'][missing_indexes_m].values)}")
    return dataframe
