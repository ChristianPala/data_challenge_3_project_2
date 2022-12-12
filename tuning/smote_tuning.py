# Library to tune the SMOTE algorithm combined with the neural network algorithm.
# Imblearn library: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
# SMOTE, Keras and generating the models we selected:
from modelling.neural_network import fit_model as fit_model_keras, create_convolutional_model, evaluate_model, \
    save_evaluation_results, create_dense_model
from modelling.trees import generate_tree_model, fit_model as fit_model_tree
from modelling.knn_logreg_naiveb_svc import create_naive_bayes_model, create_svc_model, fit_model as fit_model_ns
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
# Timing:
from tqdm import tqdm
# Global variables:
from config import scaled_datasets_path, smote_tuning_path
from modelling.train_test_validation_split import split_data

# Ensure the directory exists:
smote_tuning_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    This function creates a neural network model.
    :return: None
    """
    # Load the data:
    dataset_path = Path(scaled_datasets_path,
                        "project_2_dataset_normalized_robust_scaler_most_frequent_imputation_augmented.csv")
    df = pd.read_csv(dataset_path)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df, "default", validation=True)

    # Create over_samplers:
    sampling_strategies = [0.84, 0.85, 0.86, 0.90, 0.92, 0.93, 'auto']  # test different values
    samplers = [SVMSMOTE(sampling_strategy=sampling_strategy, random_state=42)
                for sampling_strategy in sampling_strategies]
    # samplers += [BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42)
    #                   for sampling_strategy in sampling_strategies]
    # samplers += [ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    #                   for sampling_strategy in sampling_strategies]
    # samplers += [KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=42)
    #                  for sampling_strategy in sampling_strategies]
    # samplers += [RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    #                   for sampling_strategy in sampling_strategies]
    # samplers += [RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    #                   for sampling_strategy in sampling_strategies]

    # we also had a part where we tried to combine the SMOTE with the undersampling, using a pipeline,
    # but it didn't improve, so we decided to leave it out.

    for i, sampler in tqdm(enumerate(samplers), desc="SMOTE tuning",
                           unit="sampler", total=len(samplers), colour="green"):
        # Create the model:
        model_cnn = create_convolutional_model(input_dim=x_train.shape[1])
        model_cnn_name = "convolutional_model"  # change with the model you want to use
        model_dense = create_dense_model(input_dim=x_train.shape[1])
        model_dense_name = "dense_model"
        model_gb = generate_tree_model(model_type="gradient_boosting")
        model_gb_name = "gradient_boosting"
        model_svc = create_svc_model()
        model_svc_name = "svc"
        model_nb = create_naive_bayes_model()
        model_nb_name = "naive_bayes"
        # add the models to a dictionary:
        models = {model_cnn_name: model_cnn, model_dense_name: model_dense, model_gb_name: model_gb,
                  model_svc_name: model_svc, model_nb_name: model_nb}

        for model_name, model in tqdm(models.items(), desc="Models", unit="model", total=len(models), colour="blue"):
            # Create the dataset with the pipeline or the over_sampler:
            x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)

            # clear the model weights if the model is a neural network:
            if model_name == model_cnn_name or model_name == model_dense_name:
                model.reset_states()

                # Fit the model:
                model = fit_model_keras(model, x_train_resampled, y_train_resampled)

                # Predict the target values:
                y_pred = model.predict(x_val)

            elif model_name == model_gb_name:
                model = fit_model_tree(model, x_train_resampled, y_train_resampled)
                y_pred = model.predict(x_val)

            else:
                model = fit_model_ns(model, x_train_resampled, y_train_resampled)
                y_pred = model.predict(x_val)

            # Evaluate the model:
            evaluation_results = evaluate_model(y_val, y_pred)

            # Save the evaluation results:
            save_evaluation_results(evaluation_results=evaluation_results,
                                    model_type=f'{sampler.__class__.__name__}',
                                    save_path=smote_tuning_path,
                                    dataset_name=f"{model_name}_{sampling_strategies[i % len(sampling_strategies)]}")

        # Conclusion:
        # SVM-smote is the best over-sampler for this dataset and our choices.


# Driver:
if __name__ == '__main__':
    main()
