# main file to run the project pipeline:
# Libraries:
# Main files for the project:
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
from modelling.model_evaluator import sort_all_results_by_f_1_score, evaluator_main
from tuning.balance_classes import balance_classes_main
from tuning.balanced_trees import balanced_trees_main
from tuning.balanced_neural_network import balanced_neural_network_main
from tuning.balanced_knn_logreg_naiveb_svc import balanced_other_models_main
import pprint

# Global variable to store the execution times during the pipeline:
from auxiliary.method_timer import execution_times
from config import trees_results_path, neural_networks_results_path, other_models_results_path, \
    trees_balanced_results_path, neural_networks_balanced_results_path, other_models_balanced_results_path
from tuning.evaluate_tuned_models import tuning_main


def main() -> None:
    """
    This function runs the preprocessing, feature engineering and baseline modelling for trees.
    # Todo: Expand to all the other pipelines.
    :return: None
    """
    # Preprocessing:
    # ----------------------------------------------
    preprocessor_main(suppress_print=True, missing_values_dominant_strategies=['drop', 'most_frequent_imputation'])
    feature_engineering_main(overwrite_original=True)
    scaling_main(dominant_scaling_strategies=['standard_scaler', 'robust_scaler'])
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
    balance_classes_main(dominant_strategies=['undersampled', 'oversampled', 'smote'])
    balanced_trees_main(dominant_model='gradient_boosting')
    balanced_neural_network_main()
    balanced_other_models_main(dominant_model='svc')
    evaluator_main(trees_balanced_results_path, neural_networks_balanced_results_path,
                   other_models_balanced_results_path)
    """
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
    # tuned neural network:
    # tuned other models:
    # we consider svc for tuning since it achieved the best results in the baseline and balanced models
    # tuning_main()
    # Explaining:
    # ----------------------------------------------
    # global_surrogate_main()
    # lime_main()
    # pdp_main()
    # pretty print the execution times dictionary:
    pprint.pprint(execution_times)


if __name__ == '__main__':
    main()

