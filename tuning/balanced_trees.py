# Library to test the results of the tree models with balanced data.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
# Trees
from modelling.trees import predict_model, fit_model, generate_tree_model
# Load the data:
from tuning.balanced_neural_network import load_balanced_datasets
# Evaluate the model:
from modelling.model_evaluator import evaluate_model, save_evaluation_results
# Timing:
from auxiliary.method_timer import measure_time
from tqdm import tqdm

# Global variables:
from config import balanced_datasets_path, trees_balanced_results_path


# Functions:
@measure_time
def balanced_trees_main(dominant_model: str = None) -> None:
    """
    This function runs the tree models with balanced data.
    @param dominant_model: str: the dominant model if it exists. If None, all models will be run.
    :return: None
    """
    # get the sub-folders of the balanced datasets' folders dynamically:
    sub_folder = [x.name for x in balanced_datasets_path.iterdir() if x.is_dir()]

    if not dominant_model:
        model_types: list[str] = ['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost']

        # For each sub-folder in the balanced datasets' folder:
        for sub in tqdm(sub_folder, desc="trees balanced", unit="folder", total=len(sub_folder), colour="green"):
            # Load the data:
            data = load_balanced_datasets(balanced_datasets_path / sub)
            # For each dataset:
            for train, validation, file_name in tqdm(data, desc="datasets", unit="dataset", colour="yellow"):
                # split the train and validation data:
                x_train = train.drop("default", axis=1)
                y_train = train["default"]
                x_validation = validation.drop("default", axis=1)
                y_validation = validation["default"]

                for model_name in tqdm(model_types, desc="models", unit="model", total=len(model_types), colour="blue"):
                    # Generate the model:
                    model = generate_tree_model(model_name)
                    # Fit the model:
                    fit_model(model, x_train, y_train)
                    # Predict the model:
                    y_pred = predict_model(model, x_validation)
                    # Evaluate the model:
                    evaluation = evaluate_model(y_validation, y_pred)
                    # Save the results:
                    save_evaluation_results(evaluation_results=evaluation, model_type=model_name,
                                            save_path=trees_balanced_results_path / sub / file_name,
                                            dataset_name=file_name)

    elif dominant_model == 'decision_tree':
        # For each sub-folder in the balanced datasets' folder:
        for sub in tqdm(sub_folder, desc="trees balanced", unit="folder", total=len(sub_folder), colour="green"):
            # Load the data:
            data = load_balanced_datasets(balanced_datasets_path / sub)
            # For each dataset:
            for train, validation, file_name in tqdm(data, desc="datasets", unit="dataset", colour="yellow"):
                # split the train and validation data:
                x_train = train.drop("default", axis=1)
                y_train = train["default"]
                x_validation = validation.drop("default", axis=1)
                y_validation = validation["default"]

                # Generate the model:
                model = generate_tree_model('decision_tree')
                # Fit the model:
                fit_model(model, x_train, y_train)
                # Predict the model:
                y_pred = predict_model(model, x_validation)
                # Evaluate the model:
                evaluation = evaluate_model(y_validation, y_pred)
                # Save the results:
                save_evaluation_results(evaluation_results=evaluation, model_type='decision_tree',
                                        save_path=trees_balanced_results_path / sub / file_name,
                                        dataset_name=file_name)

    elif dominant_model == 'random_forest':
        # For each sub-folder in the balanced datasets' folder:
        for sub in tqdm(sub_folder, desc="trees balanced", unit="folder", total=len(sub_folder), colour="green"):
            # Load the data:
            data = load_balanced_datasets(balanced_datasets_path / sub)
            # For each dataset:
            for train, validation, file_name in tqdm(data, desc="datasets", unit="dataset", colour="yellow"):
                # split the train and validation data:
                x_train = train.drop("default", axis=1)
                y_train = train["default"]
                x_validation = validation.drop("default", axis=1)
                y_validation = validation["default"]

                # Generate the model:
                model = generate_tree_model('random_forest')
                # Fit the model:
                fit_model(model, x_train, y_train)
                # Predict the model:
                y_pred = predict_model(model, x_validation)
                # Evaluate the model:
                evaluation = evaluate_model(y_validation, y_pred)
                # Save the results:
                save_evaluation_results(evaluation_results=evaluation, model_type='random_forest',
                                        save_path=trees_balanced_results_path / sub / file_name,
                                        dataset_name=file_name)

    elif dominant_model == 'gradient_boosting':
        # For each sub-folder in the balanced datasets' folder:
        for sub in tqdm(sub_folder, desc="trees balanced", unit="folder", total=len(sub_folder), colour="green"):
            # Load the data:
            data = load_balanced_datasets(balanced_datasets_path / sub)
            # For each dataset:
            for train, validation, file_name in tqdm(data, desc="datasets", unit="dataset", colour="yellow"):
                # split the train and validation data:
                x_train = train.drop("default", axis=1)
                y_train = train["default"]
                x_validation = validation.drop("default", axis=1)
                y_validation = validation["default"]

                # Generate the model:
                model = generate_tree_model('gradient_boosting')
                # Fit the model:
                fit_model(model, x_train, y_train)
                # Predict the model:
                y_pred = predict_model(model, x_validation)
                # Evaluate the model:
                evaluation = evaluate_model(y_validation, y_pred)
                # Save the results:
                save_evaluation_results(evaluation_results=evaluation, model_type='gradient_boosting',
                                        save_path=trees_balanced_results_path / sub / file_name,
                                        dataset_name=file_name)
