# Configuration file for the project.
# Path: config.py
# Libraries:
from pathlib import Path
# Global variables:
# ------------------------------------------------------------
project_root_path: Path = Path(__file__).parent
# Data folder:
data_path: Path = Path(project_root_path, "data")
# logs path:
logs_path: Path = Path(project_root_path, "logs")
# results path:
results_path: Path = Path(project_root_path, "results")
# Excel file path:
excel_file = Path(data_path, "Project 2 Dataset.xls")
# missing values' path:
missing_values_path = Path(data_path, "missing_values_handled")
# scaled data path:
scaled_datasets_path = Path(data_path, "scaled_datasets")
# balanced datasets path:
balanced_datasets_path = Path(data_path, "balanced_datasets")
# plots:
plot_path: Path = Path(project_root_path, "preprocessing", "plots")
# default csv:
csv_file_path = Path(missing_values_path, 'project_2_dataset_drop.csv')
# sub-folders for the baseline results:
trees_results_path: Path = Path(results_path, "trees_baseline")
neural_networks_results_path: Path = Path(results_path, "neural_network_baseline")
other_models_results_path: Path = Path(results_path, "other_models_baseline")
# sub-folders for the results of the balanced models:
trees_balanced_results_path: Path = Path(results_path, "trees_balanced")
neural_networks_balanced_results_path: Path = Path(results_path, "neural_network_balanced")
other_models_balanced_results_path: Path = Path(results_path, "other_models_balanced")
# sub-folders for the results of the tuned models:
trees_tuned_results_path: Path = Path(results_path, "trees_tuned")
neural_tuned_results_path: Path = Path(results_path, "neural_network_tuned")
other_models_tuned_results_path: Path = Path(results_path, "other_models_tuned")
# sub-folders for the model explainability folders:
# ...
# ------------------------------------------------------------