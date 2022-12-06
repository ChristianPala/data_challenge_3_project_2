# Library to perform global model-agnostic interpretations of our:
# - Convoluted Neural Network (CNN)
# - Gradient Boosting Machine (GBM)
# - Support Vector Machine Classifier (SVM)
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
# Modelling:
from keras import Model
from sklearn.metrics import f1_score
# Plotting:
import matplotlib.pyplot as plt
# Explainability:
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
from sklearn.model_selection import StratifiedKFold
# Timing:
from auxiliary.method_timer import measure_time
# Global variables:
from config import feature_permutation_results_path, partial_dependence_results_path, \
    final_test_csv_path, final_models_path
# Tensorflow logging level:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure the folders are created:
feature_permutation_results_path.mkdir(parents=True, exist_ok=True)
partial_dependence_results_path.mkdir(parents=True, exist_ok=True)


# Functions:
def permutation_feature_importance_cnn(testing: pd.DataFrame, target: str, model: Model) -> None:
    """
    Performs the permutation feature importance on keras models.
    Uses the scikit-learn inspection library: https://scikit-learn.org/stable/modules/permutation_importance.html
    @param testing: pd.DataFrame: the testing dataset.
    @param target: str: the target variable name.
    @param model: Model: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    :return: None. It plots the permutation feature importance.
    """

    # Splitting the data:
    # We are not going to modify the model, so we can look at the testing data to see how the model is
    # generalizing to unseen data:
    x_test = testing.drop(target, axis=1)
    y_test = testing[target]

    # cross validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # calculate the original f1 score with cross validation:
    original_scores = []
    for index in cv.split(x_test, y_test):
        x = x_test.iloc[index[0]]
        y = y_test.iloc[index[0]]
        predictions = model.predict(x, verbose=0)
        predictions = np.where(predictions > 0.5, 1, 0)
        original_scores.append(f1_score(y, predictions))

    original_score = np.mean(original_scores)

    # Calculate the permutation importance by iterating over the columns and scrambling the values of
    # each column and calculating the loss function:
    results_df = pd.DataFrame(columns=['feature', 'importance'])
    for col in x_test.columns:
        # use cross validation to calculate the loss function:
        scores = []
        for index in cv.split(x_test, y_test):
            x_cv = x_test.iloc[index[0]].copy()
            y_cv = y_test.iloc[index[0]].copy()
            # scramble the values of the column:
            x_cv[col] = np.random.permutation(x_cv[col])
            # calculate the loss function:
            y_pred_cv = model.predict(x_cv, verbose=0)
            y_pred_cv = np.where(y_pred_cv > 0.5, 1, 0)
            score = f1_score(y_cv, y_pred_cv)
            scores.append(score)

        # calculate the mean loss function:
        mean_score = np.mean(scores)
        # We checked this resource for how to calculate the permutation importance:
        # https://christophm.github.io/interpretable-ml-book/feature-importance.html
        # store the results:
        results_df = pd.concat([results_df,
                                pd.DataFrame({'feature': [col],
                                              'importance': [round(original_score - mean_score, 3) * 100]})], axis=0)

    # sort the results:
    results_df.sort_values(by='importance', ascending=True, inplace=True)

    # Plot the results:
    plt.figure(figsize=(10, 10))
    plt.barh(results_df['feature'], results_df['importance'])
    plt.title(f'Permutation Feature Importance {model.__class__.__name__}')
    plt.xlabel('Importance (original f1 score - 5 stratified folded mean f1 score) %')
    plt.ylabel('Feature')
    plt.subplots_adjust(left=0.3)
    plt.savefig(Path(feature_permutation_results_path,
                     f'permutation_feature_importance_{model.__class__.__name__}.png'))


def permutation_feature_importance(testing: pd.DataFrame, target: str, model: ...,
                                   random_state: int = 42) -> None:
    """
    This function carry out a Global Model-agnostic Explanation of a pre-trained model using the eli5 framework:
    https://eli5.readthedocs.io/en/latest/overview.html
    We will shuffle the values in a single column, make predictions using the resulting dataset.
    Use these predictions and the true target values to calculate how much the loss function suffered from shuffling.
    That performance deterioration measures the importance of the variable you just shuffled.
    We will go back to the original data order (undoing the previous shuffle) and repeat the procedure with the next
    column in the dataset, until we have calculated the importance of each column.
    @param testing: pd.DataFrame: the testing dataset.
    @param target: str: the target variable name.
    @param model: ...: our pre-trained black-box model, we will use it to perform a prediction and analyze it
    with our explainer.
    @param random_state: int: default = 42: the random state to be used for the split for reproducibility.
    """

    # Splitting the data:
    x_test = testing.drop(target, axis=1)
    y_test = testing[target]

    perm = PermutationImportance(model, random_state=random_state, ).fit(x_test, y_test)
    html_obj = show_weights(perm, feature_names=x_test.columns.tolist())

    # Write html object to a file (adjust file path; Windows path is used here)
    with open(Path(feature_permutation_results_path, f'permutation_feature_importance_{model.__class__.__name__}.html'),
              'wb') as f:
        f.write(html_obj.data.encode("UTF-8"))


@measure_time
def permutation_importance_main() -> None:
    """
    This function is the main function of the script. It performs the following:
    - Load the data
    - Train the model
    - Perform the global model-agnostic explanations
    @return: None
    """

    # Load the data:
    testing = pd.read_csv(final_test_csv_path)

    # load the convoluted neural network model:
    cnn_model = pd.read_pickle(Path(final_models_path, 'cnn_model.pkl'), )

    # load the gradient boosting model:
    gb_model = pd.read_pickle(Path(final_models_path, 'gradient_boosting_model.pkl'))

    # load the svm model:
    svc_model = pd.read_pickle(Path(final_models_path, 'support_vector_machine_model.pkl'))

    # perform feature permutation:
    permutation_feature_importance_cnn(testing, target='default', model=cnn_model)
    permutation_feature_importance(testing, target='default', model=gb_model)
    permutation_feature_importance(testing, target='default', model=svc_model)

    # perform partial dependence plots:
    # pass


# Driver Code:
if __name__ == '__main__':
    permutation_importance_main()
