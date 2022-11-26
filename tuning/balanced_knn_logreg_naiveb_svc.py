# Library to test the results of the other models with balanced data.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# Other models:
from modelling.knn_logreg_naiveb_svc import create_knn_model, create_logreg_model, \
    create_naive_bayes_model, create_svc_model, fit_model, predict_model
# Load the data:
from tuning.balanced_neural_network import load_balanced_datasets
# Evaluate the model:
from modelling.model_evaluator import evaluate_model, save_evaluation_results
# Timing:
from auxiliary.method_timer import measure_time
from tqdm import tqdm

# Global variables:
from config import balanced_datasets_path, other_models_balanced_results_path


# Functions:
@measure_time
def balanced_other_models_main() -> None:
    """
    This function runs the tree models with balanced data.
    :return: None
    """
    sub_folder = ["borderline_smote", "oversampled", "smote", "smote_tomek_links", "undersampled"]

    # For each sub-folder in the balanced datasets' folder:
    for sub in tqdm(sub_folder, desc="other models balanced", unit="folder", total=len(sub_folder), colour="green"):
        # Load the data:
        data = load_balanced_datasets(balanced_datasets_path / sub)
        # For each dataset:
        for train, validation, file_name in tqdm(data, desc="datasets", unit="dataset", colour="yellow"):
            # split the train and validation data:
            x_train = train.drop("default", axis=1)
            y_train = train["default"]
            x_validation = validation.drop("default", axis=1)
            y_validation = validation["default"]

            # create the models:
            knn_model = create_knn_model()
            logreg_model = create_logreg_model()
            naive_bayes_model = create_naive_bayes_model()
            svm_model = create_svc_model()

            models = [knn_model, logreg_model, naive_bayes_model, svm_model]
            model_names = ["knn", "logreg", "naive_bayes", "svc"]

            # train evaluate and save the model results:
            for models, model_names in tqdm(zip(models, model_names), desc="models", unit="model", total=len(models)):
                # Fit the model:
                fit_model(models, x_train, y_train)
                # Predict the model:
                y_pred = predict_model(models, x_validation)
                # Evaluate the model:
                evaluation = evaluate_model(y_validation, y_pred)
                # Save the results:
                save_evaluation_results(evaluation_results=evaluation, model_type=model_names,
                                        save_path=other_models_balanced_results_path / sub / file_name,
                                        dataset_name=file_name)
