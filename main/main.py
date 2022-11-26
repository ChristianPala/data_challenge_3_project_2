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
from modelling.model_evaluator import sort_all_results_by_f_1_score
from tuning.balance_classes import balance_classes_main
from tuning.balanced_trees import balanced_trees_main
from tuning.balanced_neural_network import balanced_neural_network_main
from tuning.balanced_knn_logreg_naiveb_svc import balanced_other_models_main
import pprint

# Global variable to store the execution times during the pipeline:
from auxiliary.method_timer import execution_times


def main() -> None:
    """
    This function runs the preprocessing, feature engineering and baseline modelling for trees.
    # Todo: Expand to all the other pipelines.
    :return: None
    """
    # Preprocessing:
    # ----------------------------------------------
    preprocessor_main(suppress_print=True)
    feature_engineering_main()
    scaling_main()
    eda_main()
    # Baseline models:
    # ----------------------------------------------
    trees_main()
    neural_network_main()
    other_models_main()
    sort_all_results_by_f_1_score()
    """
    Augmented datasets are usually better than the original ones.
    Supervised and unsupervised imputation are usually better than most frequent and drop.
    Scaled datasets are usually better than the original ones.
    """
    # Tuning:
    # ----------------------------------------------
    # Under-sampling , over-sampling and SMOTE variants:
    balance_classes_main()
    balanced_trees_main()
    balanced_neural_network_main()
    balanced_other_models_main()
    # tuned trees:
    # tuned neural network:
    # tuned other models:
    # Explaining:
    # ----------------------------------------------
    # ...
    # pretty print the execution times dictionary:
    pprint.pprint(execution_times)


if __name__ == '__main__':
    main()

