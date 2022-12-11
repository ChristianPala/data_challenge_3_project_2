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
    preprocessor_main(suppress_print=True, missing_values_dominant_strategies=["drop"])
    feature_engineering_main(overwrite_original=True)
    scaling_main(dominant_scaling_strategies=['robust_scaler'])
    eda_main()
    # # Baseline models:
    # # ----------------------------------------------
    trees_main()
    neural_network_main()
    other_models_main()
    evaluator_main(trees_results_path, neural_networks_results_path,
                   other_models_results_path, suppress_print=True)
    """
    Augmented vs non-augmented:
    Augmented and not augmented are similar, but the best models, which for this task are also the more complex,
    prefer the augmented data, so we will use only the augmented data.
    Dropping the missing values was usually the best strategy, the difference in performance did not justify
    the added time complexity of keeping the other strategies in the pipeline.
    Scaling:
    Robust scaling using yeo-johnson to handle skewness, and in particular negative skewness, was the best strategy.
    
    # Below an output example of the results with all possible strategies, which formed the basis of our decisions:
    # Report summary with all strategies, on which we based the selection above:
    # Note the trees are on non-preprocessed datasets:
    The best decision tree drop, with an f1 score of 0.409
    The best random forest drop augmented, with an f1 score of 0.483
    The best gradient boosting drop, with an f1 score of 0.515
    The best xgboost drop, with an f1 score of 0.477
    The best neural network convoluted normalized robust scaler most frequent imputation, with an f1 score of 0.538
    The best knn robust scaler drop, with an f1 score of 0.442
    The best logreg normalized robust scaler drop augmented, with an f1 score of 0.51
    The best naive bayes minmax scaler drop augmented, with an f1 score of 0.526
    The best svc robust scaler drop augmented, with an f1 score of 0.532
    
    After a number of trials we settled on the following best models:
    The best tree model: gradient boosting.
    The best neural network model: convoluted neural network.
    The best of the other models we tried: Support Vector Classifier.
    """
    # Tuning:
    # ----------------------------------------------
    # After tuning the balancing strategies, we selected smoteen, which was the best or
    # second best for all models.
    balance_classes_main(dominant_strategies=['smote_enn'])
    balanced_trees_main(dominant_model='gradient_boosting')
    balanced_neural_network_main(dominant_model='convolutional')
    balanced_other_models_main(dominant_model='svc')
    evaluator_main(trees_balanced_results_path, neural_networks_balanced_results_path,
                   other_models_balanced_results_path, suppress_print=True, balanced=True)
    """
    Sample output:
    The best gradient boosting robust scaler drop, with an f1 score of 0.532
    The best convolutional network robust scaler drop, with an f1 score of 0.538
    The best svc robust scaler drop, with an f1 score of 0.539
    """
    # tuned trees:
    # we consider gradient boosting for tuning since it achieved the best results in the baseline and balanced models
    # gb_main()
    # tuned neural network, we consider the convolutional neural network as it was strictly better than the dense one.
    # tuned other models:
    # cv_main()
    # we consider svc for tuning since it achieved the best results in the baseline and balanced models, naive
    # bayes was also pretty close.
    # svc_main()
    # We keep the tuning off the main pipeline, as it is very time-consuming.
    # Below the tuning of the simple models:
    # simple_models_main()
    # Below we generate the 3 best models we found during the tuning phase for the explanation part:
    create_final_models_main()
    final_comparisons_main()
    # Explaining:
    # ----------------------------------------------
    # Global:
    # ----------------------------------------------
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
