# main file to run the project pipeline:
from modelling.knn_logreg_naiveb_svc import other_models_main
from preprocessing.preprocessor import preprocessor_main
from feature_engineering.create_features import feature_engineering_main
from modelling.trees import trees_main
from preprocessing.scaling import scaling_main
from modelling.neural_network import neural_network_main


def main() -> None:
    """
    This function runs the preprocessing, feature engineering and baseline modelling for trees.
    # Todo: Expand to all the other pipelines.
    :return: None
    """
    # Preprocessing:
    preprocessor_main(suppress_print=True)
    feature_engineering_main()
    scaling_main()
    # Baseline models:
    trees_main()
    neural_network_main()
    other_models_main()
    # Tuning:
    # Explaining:


if __name__ == '__main__':
    main()

