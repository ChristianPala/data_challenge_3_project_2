# Library to tune the SMOTE algorithm combined with the neural network algorithm.
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

# Libraries:

# SMOTE and Keras:
from modelling.neural_network import fit_model, create_convolutional_model, evaluate_model, \
    save_evaluation_results, create_dense_model

# Data manipulation:
import pandas as pd
from pathlib import Path

# Global variables:
from config import scaled_datasets_path, neural_tuned_results_path
from modelling.train_test_validation_split import split_data

# Ensure the directory exists:
neural_tuned_results_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    This function creates a neural network model.
    :return: None
    """
    # Load the data:
    dataset_path = Path(scaled_datasets_path,
                        "project_2_dataset_minmax_scaler_scaling_most_frequent_imputation_augmented.csv")
    df = pd.read_csv(dataset_path)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df, "default", validation=True)

    # Create over_samplers:
    sampling_strategies = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # test different values
    samplers = [SVMSMOTE(sampling_strategy=sampling_strategy, random_state=42)
                for sampling_strategy in sampling_strategies]
    # samplers += [BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42)
    #                   for sampling_strategy in sampling_strategies]
    # samplers += [ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    #                   for sampling_strategy in sampling_strategies]
    # samplers += [KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=42)
    #                  for sampling_strategy in sampling_strategies]
    samplers += [RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                 for sampling_strategy in sampling_strategies]
    samplers += [RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                 for sampling_strategy in sampling_strategies]

    for i, sampler in tqdm(enumerate(samplers), desc="Neural network",
                           unit="sampler", total=len(samplers), colour="green"):
        # Create the model:
        model = create_dense_model(input_dim=26)

        # Create the dataset with the pipeline or the over_sampler:
        x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)

        # Fit the model:
        model = fit_model(model, x_train_resampled, y_train_resampled)

        # Predict the target values:
        y_pred = model.predict(x_val)

        # Evaluate the model:
        evaluation_results = evaluate_model(y_val, y_pred)

        # Save the evaluation results:
        save_evaluation_results(evaluation_results=evaluation_results,
                                model_type=f'{sampler.__class__.__name__}',
                                save_path=neural_tuned_results_path,
                                dataset_name=dataset_path.stem + "_" +
                                f"{sampler.__class__.__name__}_{sampling_strategies[i % len(sampling_strategies)]}")

        # Conclusion:
        # Use SVMSMOTE for balancing the neural network model.
        # Implemented.


# Driver:
if __name__ == '__main__':
    main()
