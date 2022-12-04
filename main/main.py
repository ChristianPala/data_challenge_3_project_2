# main file to run the project pipeline:
# Libraries:
# Main files for the project:
from model_explainability.create_final_models import create_final_models_main
from model_explainability.global_surrogate import global_surrogate_main
from model_explainability.local_interpretable_model_agnostic_explanations import lime_main
from model_explainability.partial_dependece_plot import pdp_main
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
from tuning.evaluate_tuned_models import tuning_main

import pprint

# Global variable to store the execution times during the pipeline:
from auxiliary.method_timer import execution_times
from config import trees_results_path, neural_networks_results_path, other_models_results_path, \
    trees_balanced_results_path, neural_networks_balanced_results_path, other_models_balanced_results_path


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
    Drop is the best for non-neural networks, so we keep it. Most frequent is the best for neural networks, 
    so we keep it. 
    Scaling:
    Standard and Robust, with our skeweness handling, are the best, so we keep them. We keep both normalized and
    non-normalized.
    
    # Report summary with all strategies, on which we based the selection above:
    The best decision tree drop, with an f1 score of 0.409
    The best random forest drop augmented, with an f1 score of 0.483
    The best gradient boosting drop, with an f1 score of 0.515
    The best xgboost drop, with an f1 score of 0.477
    The best neural network convoluted normalized standard scaler most frequent imputation, with an f1 score of 0.538
    The best knn robust scaler drop, with an f1 score of 0.442
    The best logreg normalized robust scaler drop augmented, with an f1 score of 0.51
    The best naive bayes minmax scaler drop augmented, with an f1 score of 0.526
    The best svc robust scaler drop augmented, with an f1 score of 0.532
    """
    # Tuning:
    # ----------------------------------------------
    # Under-sampling , over-sampling and SMOTE variants:
    # oversampled is the best for the neural network, undersampled for the svc and
    # smote tomek for gradient boosting.
    balance_classes_main(dominant_strategy=['undersampled', 'oversampled', 'smote_tomek'])
    balanced_trees_main(dominant_model='gradient_boosting')
    balanced_neural_network_main(dominant_model='convolutional')
    balanced_other_models_main(dominant_model='svc')
    evaluator_main(trees_balanced_results_path, neural_networks_balanced_results_path,
                   other_models_balanced_results_path, suppress_print=True, balanced=True)
    """
    Based on the balancing results we selected:
    Gradient boosting as the best model for trees
    Convolutional neural network as the best model for neural networks
    SVC as the best model for the other models
    # Sample output:
    The best decision tree minmax scaler drop, with an f1 score of 0.455
    The best random forest minmax scaler drop, with an f1 score of 0.539
    The best gradient boosting minmax scaler drop, with an f1 score of 0.545
    The best xgboost minmax scaler drop, with an f1 score of 0.531
    The best neural network minmax scaler most frequent imputation, with an f1 score of 0.529
    The best knn minmax scaler drop, with an f1 score of 0.504
    The best logreg minmax scaler drop, with an f1 score of 0.532
    The best naive bayes minmax scaler drop, with an f1 score of 0.525
    The best svc minmax scaler drop, with an f1 score of 0.541
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
    # Below we generate the 3 best models we found.
    create_final_models_main()
    # Explaining:
    # ----------------------------------------------
    # feature permutation:
    # dependence plots:
    # global_surrogate_main()
    # lime_main()
    # pretty print the execution times dictionary:
    pprint.pprint(execution_times)


if __name__ == '__main__':
    main()
