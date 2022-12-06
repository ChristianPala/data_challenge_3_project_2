# Libraries
# Data manipulation
import pickle

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
# from global_model_agnostic_explanations import *
from global_surrogate import *
from partial_dependece_plot import *

from keras.models import load_model


def xai_main():
    # Testing functions
    # nn = load_model(Path(final_models_path, 'neural_network_model.h5'))

    # training = pd.read_csv(final_training_csv_path)
    # x_train = training.drop("default", axis=1)
    # y_train = training["default"]
    #
    # testing = pd.read_csv(final_training_csv_path)
    # x_test = testing.drop("default", axis=1)
    # y_test = testing["default"]

    # # generate the model:
    # model = generate_tree_model(model_type='random_forest')
    # # fit the model:
    # model = fit_model(model=model, x_train=x_train, y_train=y_train)


    # global_surrogate_main()
    # pdp_main()
    # bho = model.predict_proba(x_test)
    #
    # prob = nn.predict(x_test)
    # my_list = map(lambda x: x[0], prob)
    # prob = pd.Series(my_list)
    # nprob = prob.apply(lambda x: 1-x)
    #
    # feature_matrix = np.column_stack((prob, nprob))

    # lime_explanation(training, testing, 'default', nn, 375, False, True)
    # lime_explanation(training, testing, 'default', model, 0, True)
    # shap_explanation(training, testing, 'default', model, "tree", 450, True)
    # testing = pd.read_csv(final_training_csv_path)[:500]

    # shap_explanation(training, testing, 'default', nn, "kernel", "nn", 375, True)
    # permutation_feature_importance(training, testing, 'default', nn)
    # partial_dependence_plots(training, testing, 'default', nn)

    training = pd.read_csv(final_training_oversampled_csv_path)
    validation = pd.read_csv(final_validation_oversampled_csv_path)
    testing = pd.read_csv(final_testing_oversampled_csv_path)
    training = pd.concat([training, validation])

    # Explaining surrogate model worst predictions
    bb = load_model(Path(global_surrogate_models_path, 'black_box_model.h5'))
    bb = load_model(Path(final_neural_network_path))
    lr = pickle.load(open(Path(global_surrogate_models_path, 'logistic_regression_surrogate_model.pkl'), 'rb'))
    rf = pickle.load(open(Path(global_surrogate_models_path, 'random_forest_surrogate_model.pkl'), 'rb'))

    worst_surrogate_obs = [984, 306, 531, 4336, 4058, 851]

    # for o in worst_surrogate_obs:
    #     lime_explanation(training, testing, 'default', bb, 'black_box', o, False, False)
    #     lime_explanation(training, testing, 'default', lr, 'log_reg', o, True, False)
    #     lime_explanation(training, testing, 'default', rf, 'random_f', o, True, False)

    # Local explanations of our best final models
    # nn = load_model(Path(final_neural_network_path))
    # gb = pickle.load(open(Path(final_models_path, 'gradient_boosting_model.pkl'), 'rb'))
    svc = pickle.load(open(Path(final_models_path, 'supper_vector_machine_model.pkl'), 'rb'))

    j = 2009

    # lime_explanation(training, testing, 'default', nn, 'black_box', j, False, False)
    # lime_explanation(training, testing, 'default', gb, 'log_reg', j, True, False)
    lime_explanation(training, testing, 'default', svc, 'random_f', j, True, False)

    # shap_explanation(training, testing, 'default', nn, "kernel", "nn", j, False)
    # shap_explanation(training, testing, 'default', gb, "kernel", "nn", j, False)
    shap_explanation(training, testing, 'default', svc, "kernel", "svc", j, False)


if __name__ == '__main__':
    xai_main()
