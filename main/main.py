# main file to run the project pipeline:
# Libraries:
# Main files for the project:

from preprocessing.eda.EDA import eda_main
from preprocessing.preprocessor import preprocessor_main
from feature_engineering.create_features import feature_engineering_main
from preprocessing.scaling import scaling_main
from modelling.trees import trees_main
from modelling.neural_network import neural_network_main
from modelling.knn_logreg_naiveb_svc import other_models_main
from modelling.model_evaluator import evaluator_main
from tuning.balance_classes import balance_classes_main
from tuning.balanced_trees import balanced_trees_main
from tuning.balanced_neural_network import balanced_neural_network_main
from tuning.balanced_knn_logreg_naiveb_svc import balanced_other_models_main
from model_explainability.white_box_model import white_box_main
from model_explainability.create_final_models import create_final_models_main
from model_explainability.global_surrogate import global_surrogate_main
from model_explainability.global_model_interpration_with_shap import global_shap_main
from model_explainability.local_interpretable_model_agnostic_explanations import lime_and_shap_main
from model_explainability.partial_dependece_plot import pdp_main
from model_explainability.permutation_importance import permutation_importance_main
from final_models_comparison.comparison_plots import final_comparisons_main

import pprint

# Global variable to store the execution times during the pipeline:
from auxiliary.method_timer import execution_times
from config import trees_results_path, neural_networks_results_path, other_models_results_path, \
    trees_balanced_results_path, neural_networks_balanced_results_path, other_models_balanced_results_path
from tuning.tune_simple_models import simple_models_main


def main() -> None:
    """
    This function runs the preprocessing, feature engineering and baseline modelling for trees.
    :return: None
    """
    # Preprocessing:
    # ----------------------------------------------
    preprocessor_main(suppress_print=True,
                      missing_values_dominant_strategies=['most_frequent_imputation', 'unsupervised_imputation'])
    feature_engineering_main(overwrite_original=True)
    scaling_main(dominant_scaling_strategies=['robust_scaler'], save_non_normalized=False)
    eda_main()
    # # Baseline models:
    # # ----------------------------------------------
    trees_main()
    neural_network_main()
    other_models_main()
    evaluator_main(trees_results_path, neural_networks_results_path,
                   other_models_results_path, suppress_print=True)

    # Tuning:
    # ----------------------------------------------
    balance_classes_main(dominant_strategies=["undersampled", "oversampled", "svmsmote"])
    balanced_trees_main(dominant_model='gradient_boosting')
    balanced_neural_network_main()
    balanced_other_models_main(dominant_model='naive_bayes')
    balanced_other_models_main(dominant_model='svc')
    evaluator_main(trees_balanced_results_path, neural_networks_balanced_results_path,
                   other_models_balanced_results_path, suppress_print=True, balanced=True)

    # Tuning:
    # ----------------------------------------------
    # gb_main()
    # tuned neural network, we consider the dense model as the best one
    # tuned other models:
    # cv_main()
    # we consider svc for tuning since it achieved the best results in the baseline and balanced models, naive
    # bayes was also pretty close.
    # svc_main()
    # We keep the tuning off the main pipeline, as it is very time-consuming.
    # Below the tuning of the simple models:
    # simple_models_main()
    # Below we generate the 3 best models we found during the tuning phase for the explanation part:
    # create_final_models_main()
    # final_comparisons_main()
    # Explaining:
    # ----------------------------------------------
    # Global:
    # ----------------------------------------------
    # white box:
    # white_box_main()
    # feature permutation:
    # permutation_importance_main()
    # dependence plots:
    # pdp_main()
    # global surrogate:
    # global_surrogate_main()
    # global shap:
    # global_shap_main()
    # Local:
    # ----------------------------------------------
    # Lime and Shap:
    # lime_and_shap_main()
    # ----------------------------------------------
    # pretty print the execution times dictionary:
    # Due to the long execution time we saved th results of the tuning and explainability mains in the results' folder.
    # All modules can be run independently, as long as the main pipeline is run first.
    pprint.pprint(execution_times)
    # Done:
    # ------------------------------------------------


if __name__ == '__main__':
    main()
