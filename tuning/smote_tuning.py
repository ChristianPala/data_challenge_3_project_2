# Library to tune the SMOTE algorithm combined with the neural network algorithm.
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE

# Libraries:

# SMOTE and Keras:
from modelling.neural_network import fit_model, create_convolutional_model, evaluate_model, \
    save_evaluation_results

# Data manipulation:
import pandas as pd
from pathlib import Path

# Global variables:
from config import scaled_datasets_path, neural_tuned_results_path
from modelling.train_test_validation_split import split_data

neural_tuned_results_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    This function creates a neural network model.
    :return: None
    """
    # Load the data:
    df = pd.read_csv(Path(scaled_datasets_path, "project_2_dataset_robust_scaler_scaling_drop_augmented.csv"))

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df, "default", validation=True)

    # Create over_samplers:
    over_samplers = [SMOTE(sampling_strategy=0.4, random_state=42),
                     BorderlineSMOTE(sampling_strategy=0.4, random_state=42),
                     SVMSMOTE(sampling_strategy=0.4, random_state=42),
                     ADASYN(sampling_strategy=0.4, random_state=42),
                     KMeansSMOTE(sampling_strategy=0.4, random_state=42),
                     SMOTEENN(sampling_strategy=0.4, random_state=42),
                     SMOTETomek(sampling_strategy=0.4, random_state=42)]

    for over_sampler in over_samplers:
        # Create the model:
        model = create_convolutional_model(input_dim=26)

        # Create the dataset:
        x_train_resampled, y_train_resampled = over_sampler.fit_resample(x_train, y_train)

        # Fit the model:
        model = fit_model(model, x_train_resampled, y_train_resampled)

        # Predict the target values:
        y_pred = model.predict(x_val)

        # Evaluate the model:
        evaluation_results = evaluate_model(y_val, y_pred)

        # Save the evaluation results:
        save_evaluation_results(evaluation_results=evaluation_results,
                                model_type=f'{over_sampler.__class__.__name__}',
                                save_path=neural_tuned_results_path,
                                dataset_name=f"project_2_dataset_robust_scaler_scaling_drop_augmented_balanced"
                                             f"{over_sampler.__class__.__name__}")


if __name__ == '__main__':
    main()
