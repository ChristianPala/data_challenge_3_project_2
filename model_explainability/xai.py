# Libraries
# Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Modelling
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from modelling.train_test_validation_split import split_data

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report

# Timing:
from auxiliary.method_timer import measure_time
# Global variables
from config import *
if not trees_results_path.exists():
    trees_results_path.mkdir(parents=True)

from modelling.trees import *
from local_model_agnostic_explanations import *
from global_model_agnostic_explanations import *

from keras.models import load_model


def xai_main():
    model = load_model(Path(final_models_path, 'neural_network_model.h5'))

    # df = pd.read_csv('../data/project_2_dataset_unsupervised_imputation_augmented.csv')
    # x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)
    # # generate the model:
    # model = generate_tree_model(model_type='random_forest')
    # # fit the model:
    # model = fit_model(model=model, x_train=x_train, y_train=y_train)
    training = pd.read_csv(final_training_csv_path)
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    testing = pd.read_csv(final_training_csv_path)
    x_test = testing.drop("default", axis=1)
    y_test = testing["default"]

    lime_explanation(training, testing, 'default', model, 89, True)
    shap_explanation(training, testing, 'default', model, 89, True)
    permutation_feature_importance(training, testing, 'default', model)
    partial_dependence_plots(training, testing, 'default', model)


if __name__ == '__main__':
    xai_main()
