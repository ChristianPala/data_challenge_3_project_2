# Library to test the results of the neural network with balanced data.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# Neural network:
from modelling.neural_network import predict_model, fit_model, \
    create_convolutional_model, create_dense_model
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
            train = pd.read_csv(Path(file, "final_training.csv"))
            validation = pd.read_csv(Path(file, "final_validation.csv"))
            # Return the data:
            yield train, validation, file.name
        else:
            continue
        data.append((train, validation, file.name))

    return data


@measure_time
def balanced_neural_network_main(dominant_model: [str] = None) -> None:
    """
    This function runs the neural network with balanced data.
    @param dominant_model: str: the dominant model if it exists. If None, all models will be run.
    :return: None
    """
    # get the sub-folders of the balanced datasets' folders dynamically:
    sub_folder = [x.name for x in balanced_datasets_path.iterdir() if x.is_dir()]

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
            if dominant_model == "convolutional":
                models = [create_convolutional_model(x_train.shape[1])]
            elif dominant_model == "dense":
                models = [create_dense_model(x_train.shape[1])]
            else:
                model_c = create_convolutional_model(x_train.shape[1])
                model_d = create_dense_model(x_train.shape[1])
                models = [model_c, model_d]
            # Fit the model:
            if len(models) == 1:
                fit_model(models[0], x_train, y_train)
                # Predict the model:
                y_pred = predict_model(models[0], x_validation)
                # Evaluate the model:
                evaluation_results = evaluate_model(y_validation, y_pred)
                # Save the model:
                models_path: Path = Path(neural_networks_balanced_results_path, "models")
                models_path.mkdir(parents=True, exist_ok=True)
                models[0].save(Path(models_path, f"{file_name}.h5"))
                # Save the results:
                save_evaluation_results(evaluation_results=evaluation_results, model_type=dominant_model + "_network",
                                        save_path=neural_networks_balanced_results_path / sub / file_name,
                                        dataset_name=file_name)
            else:
                for i, model in enumerate(models):
                    dominant_model = "convolutional" if i == 0 else "dense"
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
                    save_evaluation_results(evaluation_results=evaluation_results, model_type=dominant_model
                                                                                              + "_network",
                                            save_path=neural_networks_balanced_results_path / sub / file_name,
                                            dataset_name=file_name)


# Driver:
if __name__ == '__main__':
    balanced_neural_network_main()
