# We have a strong class imbalance in our data. We will use undersampling, oversampling
# and SMOTE to balance the classes and see if it improves the performance of our baseline models.

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
import shutil

from auxiliary.method_timer import measure_time
# Modelling:
from modelling.train_test_validation_split import split_data
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

# Global variables:
# Path to the data folder:
data_path = Path("..", "data")
balanced_datasets_path = Path(data_path, "balanced_training_datasets")
# Path to the missing_values_handled folder:
missing_values_path = Path(data_path, "missing_values_handled")
# Path to the scaled_datasets folder:
scaled_datasets_path = Path(data_path, "scaled_datasets")


# Functions:
def show_balance(dataframe: pd.DataFrame) -> None:
    """
    Show the balance of the classes in the dataset.
    @param dataframe: pd.DataFrame: the dataframe containing the dataset.
    @return: None
    """
    # Show the balance of the classes:
    print(dataframe["default"].value_counts(normalize=True))
    # we have a 78% - 22% class imbalance.


def undersample_transform(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Transformation using the imbalance learn implementation of Random under-sampling:
    https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: pd.Dataframe: the balanced dataframe.
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    x = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our under-sampler
    under = RandomUnderSampler(sampling_strategy="auto", random_state=42)

    # Transform the dataset
    x, y = under.fit_resample(x, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    # merge the target and the features:
    df = pd.DataFrame(pd.concat([x, y], axis=1))

    return df


def oversample_transform(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Transformation using the imbalance learn implementation of Random over-sampling:
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: pd.Dataframe: the balanced dataframe.
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    x = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our over-sampler
    over = RandomOverSampler(sampling_strategy="auto", random_state=42)

    # Transform the dataset
    x, y = over.fit_resample(x, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    # merge the target and the features:
    df = pd.DataFrame(pd.concat([x, y], axis=1))

    return df


def smote_transform(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Transformation using the imbalance learn implementation of SMOTE(synthetic minority over-sampling technique):
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    The original paper on SMOTE suggested combining SMOTE with random under-sampling of the majority class.:
    https://arxiv.org/abs/1106.1813
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: pd.Dataframe: the balanced dataframe
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    x = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our over-sampler and under-sampler
    over = SMOTE(sampling_strategy="auto", random_state=42, n_jobs=-1)

    # Transform the dataset
    x, y = over.fit_resample(x, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    # merge the target feature with the rest of the features
    df = pd.DataFrame(pd.concat([x, y], axis=1))

    return df


def borderline_smote(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Transformation using the imbalance learn implementation of Borderline SMOTE:
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: pd.Dataframe: the balanced dataframe
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    X = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our over-sampler
    over = BorderlineSMOTE(sampling_strategy="auto", random_state=42, n_jobs=-1)

    # Transform the dataset
    x, y = over.fit_resample(X, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    # merge the target feature with the rest of the features
    df = pd.DataFrame(pd.concat([x, y], axis=1))

    return df


def smote_tomek(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Transformation using the imbalance learn implementation of SMOTE + Tomek links:
    https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html
    :param train: the dataframe to be analyzed
    :param target: the column name of the target feature
    :return: pd.Dataframe: the balanced dataframe
    """
    df = train.copy()

    # We split the target and the rest of the features
    y = df[target]
    X = df.drop([target], axis=1)

    # Check the current balance of the target feature
    # counter = collections.Counter(y)
    # print(counter)

    # Defining our over-sampler
    over = SMOTETomek(sampling_strategy="auto", random_state=42, n_jobs=-1)

    # Transform the dataset
    x, y = over.fit_resample(X, y)

    # Check the balance of the target feature after the resample
    # counter = collections.Counter(y)
    # print(counter)

    # merge the target feature with the rest of the features
    df = pd.DataFrame(pd.concat([x, y], axis=1))

    return df


def save_dataframes(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    """
    Save the dataframes to disk, auxiliary function to perform_balancing.
    @param train: the training dataframe
    @param val: the validation dataframe
    @param test: the test dataframe
    @param path: the path to save the dataframes
    :return: None
    """
    # ensure the path exists
    path.mkdir(parents=True, exist_ok=True)

    # save the dataframes
    train.to_csv(path / "train.csv", index=False)
    val.to_csv(path / "val.csv", index=False)
    test.to_csv(path / "test.csv", index=False)


def perform_balancing(tr: pd.DataFrame, vl: pd.DataFrame, tst: pd.DataFrame, target: str, method: str,
                      file_stem: str) -> None:
    """
    Perform the balancing of the dataframes and save them to disk.
    @param tr: the training dataframe
    @param vl: the validation dataframe
    @param tst: the test dataframe
    @param target: the target feature's name
    @param method: the balancing method to use, one of:
    - undersampled,
    - oversampled,
    - smote,
    - borderline_smote,
    - smote_tomek_links
    @param file_stem: the path to save the dataframes to.
    :return: None
    """

    # perform the rebalancing
    if method == "undersampled":
        tr = undersample_transform(tr, target)
    elif method == "oversampled":
        tr = oversample_transform(tr, target)
    elif method == "smote":
        tr = smote_transform(tr, target)
    elif method == "borderline_smote":
        tr = borderline_smote(tr, target)
    elif method == "smote_tomek_links":
        tr = smote_tomek(tr, target)
    else:
        raise ValueError(f"Unknown method {method}")

    # remove project_2_dataset from the file stem
    file_stem = file_stem.replace("project_2_dataset_", "")

    # save the dataframes
    save_dataframes(tr, vl, tst, Path(f"../data/balanced_training_datasets/{method}/{file_stem}"))


@measure_time
def balance_classes_main(suppress_print=True) -> None:
    """
    Saves the balanced datasets with over-, under-sampling and SMOTE
    @param suppress_print: bool: whether to suppress the print statements.
    @return: None. Saves the csv files in the balanced_training_datasets folder.
    """
    # get all the files in the scaling folder, since it performed better than the non-scaled data:
    csv_files = list(scaled_datasets_path.glob("*.csv"))
    # remove the "drop" and "most_frequent" strategies since the other missing value strategies performed better in
    # all cases:
    csv_files = [file for file in csv_files if "drop" not in file.name and "most_frequent" not in file.name]

    # clean the balanced_training_datasets folder:
    if balanced_datasets_path.exists() and balanced_datasets_path.is_dir():
        shutil.rmtree(balanced_datasets_path)
    balanced_datasets_path.mkdir(exist_ok=True)

    for file in csv_files:
        # read the csv file:
        dataframe = pd.read_csv(file)
        # Split the data:
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(dataframe, "default", validation=True)
        training_dataframe = pd.DataFrame(pd.concat([x_train, y_train], axis=1))
        validation_dataframe = pd.DataFrame(pd.concat([x_val, y_val], axis=1))
        test_dataframe = pd.DataFrame(pd.concat([x_test, y_test], axis=1))

        # perform the balancing with all the methods:
        methods = ["undersampled", "oversampled", "smote", "borderline_smote", "smote_tomek_links"]
        for method in methods:
            perform_balancing(training_dataframe, validation_dataframe, test_dataframe, "default", method, file.stem)
            if not suppress_print:
                print(f"Finished balancing with {method} for {file.stem}")


if __name__ == "__main__":
    balance_classes_main()
