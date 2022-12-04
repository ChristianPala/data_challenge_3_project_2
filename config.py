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
eda_csv_file_path = Path(missing_values_path, 'project_2_dataset_drop_augmented.csv')
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
undersampled_datasets_path: Path = Path(balanced_datasets_path, "undersampled")
# sub-folders for the model explainability folders:
model_explainability: Path = Path(project_root_path, "model_explainability")
final_models_path: Path = Path(model_explainability, "final_models")

final_training_undersampled_csv_path: Path = Path(balanced_datasets_path, "undersampled", "robust_scaler_scaling_drop", "final_training.csv")
final_validation_undersampled_csv_path: Path = Path(balanced_datasets_path, "undersampled", "robust_scaler_scaling_drop", "final_validation.csv")
final_test_undersampled_csv_path: Path = Path(balanced_datasets_path, "undersampled", "robust_scaler_scaling_drop", "final_test.csv")

final_training_oversampled_csv_path: Path = Path(balanced_datasets_path, "oversampled", "robust_scaler_scaling_drop", "final_training.csv")
final_validation_oversampled_csv_path: Path = Path(balanced_datasets_path, "oversampled", "robust_scaler_scaling_drop", "final_validation.csv")
final_test_oversampled_csv_path: Path = Path(balanced_datasets_path, "oversampled", "robust_scaler_scaling_drop", "final_test.csv")

final_test_tomek_csv_path: Path = Path(balanced_datasets_path, "smote_tomek_links", "robust_scaler_scaling_drop", "final_testing.csv")
final_train_tomek_csv_path: Path = Path(balanced_datasets_path, "smote_tomek_links", "robust_scaler_scaling_drop", "final_training.csv")
final_val_tomek_csv_path: Path = Path(balanced_datasets_path, "smote_tomek_links", "robust_scaler_scaling_drop", "final_validation.csv")

final_neural_network_path: Path = Path(final_models_path, "neural_network_model.h5")
global_surrogate_results_path: Path = Path(results_path, "global_surrogate_results")
partial_dependence_results_path: Path = Path(results_path, "partial_dependence_results")
# ------------------------------------------------------------
