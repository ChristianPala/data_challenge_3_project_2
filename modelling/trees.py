# Modelling with decision trees, random forests and gradient boosting
# Libraries
# Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
# Modelling
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tqdm import tqdm
from xgboost import XGBClassifier
from modelling.model_evaluator import save_evaluation_results, evaluate_model
from modelling.train_test_validation_split import split_data
# Timing:
from auxiliary.method_timer import measure_time

# Global variables
from config import missing_values_path, trees_results_path
if not trees_results_path.exists():
    trees_results_path.mkdir(parents=True)


def generate_tree_model(model_type: str) -> BaseEstimator:
    """
    This function generates one of the tree models.
    @param model_type: str: the type of the model to be generated. Supported:
     - 'decision_tree',
     - 'random_forest',
     - 'gradient_boosting'
     - 'xgboost'.
    :return: Model object.
    """
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(random_state=42)
    elif model_type == 'random_forest':
        return RandomForestClassifier(random_state=42)
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=42)
    elif model_type == 'xgboost':
        return XGBClassifier(random_state=42)
    else:
        raise ValueError('The model type is not supported.')


def fit_model(model: BaseEstimator, x_train: np.array, y_train: np.array) -> BaseEstimator:
    """
    This function fits the model to the training data.
    @param model: BaseEstimator: the model to be fitted.
    @param x_train: np.array: the training data.
    @param y_train: np.array: the target data.
    :return: Fitted model object.
    """
    model.fit(x_train, y_train)
    return model


def predict_model(model: BaseEstimator, x_test: np.array) -> np.array:
    """
    This function predicts the target values for the test data.
    @param model: BaseEstimator: the model to be used for prediction.
    @param x_test: np.array: the test data.
    :return: Predicted target values as np.array.
    """
    return model.predict(x_test)


@measure_time
def trees_main() -> None:
    """
    Main function for the baseline results on decision trees, random forests, gradient boosting and
    xgboost classifiers on the project 2 dataset.
    @return: None, saves the results in the results' folder.
    """
    # get all the csv files in the missing_values_handled folder
    if not missing_values_path.exists():
        raise ValueError('The missing values file does not exist. Cannot continue.')
    csv_files: list[Path] = list(missing_values_path.glob('*.csv'))
    model_types: list[str] = ['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost']

    if trees_results_path.exists() and trees_results_path.is_dir():
        shutil.rmtree(trees_results_path, ignore_errors=True)
    trees_results_path.mkdir(exist_ok=True, parents=True)

    for csv_file in tqdm(csv_files, desc='Tree models', unit='file', total=len(csv_files), colour='green'):
        # read the data:
        df = pd.read_csv(csv_file)
        # get the preprocessing steps from the name:
        preprocessing_steps: list[str] = csv_file.stem.split('_')[3:]
        # split the data:
        x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)
        # loop through the model types:
        for model_type in tqdm(model_types, desc='Model types', unit='model', total=len(model_types), colour='blue'):
            # generate the model:
            model = generate_tree_model(model_type=model_type)
            # fit the model:
            model = fit_model(model=model, x_train=x_train, y_train=y_train)
            # predict the target values:
            y_pred = predict_model(model=model, x_test=x_val)
            # evaluate the model:
            evaluation_results = evaluate_model(y_test=y_val, y_pred=y_pred)
            # save the results:
            save_evaluation_results(evaluation_results=evaluation_results, model_type=model_type,
                                    save_path=trees_results_path, dataset_name='_'.join(preprocessing_steps))


if __name__ == '__main__':
    trees_main()
