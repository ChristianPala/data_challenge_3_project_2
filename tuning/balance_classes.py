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


@measure_time
def balance_classes_main(suppress_print=True) -> None:
    """
    Saves the balanced datasets with over-, under-sampling and SMOTE
    @param suppress_print: bool: whether to suppress the print statements.
    @return: None. Saves the csv files in the balanced_training_datasets folder.
    """
    # get all the files in the missing_values_handled and scaling folders:
    csv_files = list(missing_values_path.glob("*.csv")) + list(scaled_datasets_path.glob("*.csv"))

    # clean the balanced_training_datasets folder:
    balanced_datasets_path = Path(data_path, "balanced_training_datasets")
    if balanced_datasets_path.exists() and balanced_datasets_path.is_dir():
        shutil.rmtree(balanced_datasets_path)
    balanced_datasets_path.mkdir(exist_ok=True)

    for file in csv_files:
        # read the csv file:
        dataframe = pd.read_csv(file)
        # Split the data:
        x_train, _, _, y_train, _, _ = split_data(dataframe, "default", validation=True)
        training_dataframe = pd.DataFrame(pd.concat([x_train, y_train], axis=1))
        # Show the balance of the classes if not suppressed:
        if not suppress_print:
            print(f"Balance of classes in {file.name} before balancing:")
            show_balance(training_dataframe)
        # Under-sample the data:
        under_sampled: pd.DataFrame = undersample_transform(training_dataframe, "default")
        # Show the balance of the classes if not suppressed:
        if not suppress_print:
            print(f"Balance of classes in {file.name} after undersampling:")
            show_balance(under_sampled)
        # Save the undersampled data:
        # Path to the undersampled data:
        undersampled_path = Path(balanced_datasets_path, "undersampled")
        undersampled_path.mkdir(exist_ok=True)
        under_sampled.to_csv(Path(undersampled_path, f"training_under_sampled_{file.name}"), index=False)

        # Over-sample the data:
        over_sampled: pd.DataFrame = oversample_transform(training_dataframe, "default")
        if not suppress_print:
            print(f"Balance of classes in {file.name} after oversampling:")
            show_balance(over_sampled)
        # Save the oversampled data:
        # Path to the oversampled data:
        oversampled_path = Path(balanced_datasets_path, "oversampled")
        oversampled_path.mkdir(exist_ok=True)
        over_sampled.to_csv(Path(oversampled_path, f"training_over_sampled_{file.name}"), index=False)

        # SMOTE the data:
        smote: pd.DataFrame = smote_transform(training_dataframe, "default")
        # Show the balance of the classes:
        if not suppress_print:
            print(f"Balance of classes in {file.name} after SMOTE:")
            show_balance(smote)
        # Save the SMOTE data:
        # Path to the SMOTE data:
        smote_path = Path(balanced_datasets_path, "smote")
        smote_path.mkdir(exist_ok=True)
        smote.to_csv(Path(smote_path, f"training_smote_{file.name}"), index=False)

        # Borderline SMOTE the data:
        borderline_smote_df: pd.DataFrame = borderline_smote(training_dataframe, "default")
        # Show the balance of the classes:
        if not suppress_print:
            print(f"Balance of classes in {file.name} after borderline SMOTE:")
            show_balance(borderline_smote_df)
        # Save the borderline SMOTE data:
        # Path to the borderline SMOTE data:
        borderline_smote_path = Path(balanced_datasets_path, "borderline_smote")
        borderline_smote_path.mkdir(exist_ok=True)
        borderline_smote_df.to_csv(Path(borderline_smote_path, f"training_borderline_smote_{file.name}"), index=False)

        # SMOTE Tomek links the data:
        smote_tomek_links: pd.DataFrame = smote_tomek(training_dataframe, "default")
        # Show the balance of the classes:
        if not suppress_print:
            print(f"Balance of classes in {file.name} after SMOTE Tomek links:")
            show_balance(smote_tomek_links)
        # Save the SMOTE Tomek links data:
        # Path to the tomek_links_smote data:
        tomek_links_smote_path = Path(balanced_datasets_path, "tomek_links_smote")
        tomek_links_smote_path.mkdir(exist_ok=True)
        smote_tomek_links.to_csv(Path(tomek_links_smote_path, f"training_smote_tomek_links_{file.name}"), index=False)


if __name__ == "__main__":
    balance_classes_main()

