# Library to test the results of the neural network with balanced data.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# Neural network:
from modelling.neural_network import predict_model, fit_model, \
    create_model
# Evaluate the model:
from modelling.model_evaluator import evaluate_model, save_evaluation_results
# Timing:
from auxiliary.method_timer import measure_time
from tqdm import tqdm

# Global variables:
from config import balanced_datasets_path, neural_networks_balanced_results_path


# Functions:
def load_balanced_datasets(folder: Path) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    This function loads the training, validation and testing data from the given folder.
    @param folder: Path: the folder containing the data.
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame: the training, validation and testing data.
    """
    data: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []

    # For each folder in the given folder:
    for file in folder.iterdir():
        # If the file is a folder:
        if file.is_dir():
            # Load the data:
            train = pd.read_csv(Path(file, "train.csv"))
            validation = pd.read_csv(Path(file, "val.csv"))
            # Return the data:
            yield train, validation, file.name
        else:
            continue
        data.append((train, validation, file.name))

    return data


@measure_time
def balanced_neural_network_main() -> None:
    """
    This function runs the neural network with balanced data.
    :return: None
    """
    sub_folder = ["borderline_smote", "oversampled", "smote", "smote_tomek_links", "undersampled"]

    # For each sub-folder in the balanced datasets' folder:
    for sub in tqdm(sub_folder, desc="Balanced neural networks", unit="folder", total=len(sub_folder), colour="green"):
        # Load the data:
        data = load_balanced_datasets(balanced_datasets_path / sub)
        # For each dataset:
        for train, validation, file_name in tqdm(data, desc="datasets", unit="dataset", colour="yellow"):
            # split the train and validation data:
            x_train = train.drop("default", axis=1)
            y_train = train["default"]
            x_validation = validation.drop("default", axis=1)
            y_validation = validation["default"]
            # Create the model:
            model = create_model(input_dim=x_train.shape[1])
            # Fit the model:
            fit_model(model, x_train, y_train)
            # Predict the model:
            y_pred = predict_model(model, x_validation)
            # Evaluate the model:
            evaluation_results = evaluate_model(y_validation, y_pred)
            # Save the model:
            models_path: Path = Path(neural_networks_balanced_results_path, "models")
            models_path.mkdir(parents=True, exist_ok=True)
            model.save(Path(models_path, f"{file_name}.h5"))
            # Save the results:
            save_evaluation_results(evaluation_results=evaluation_results, model_type="neural_network",
                                    save_path=neural_networks_balanced_results_path,
                                    dataset_name=sub + "_" + file_name)


# Driver:
if __name__ == '__main__':
    balanced_neural_network_main()
