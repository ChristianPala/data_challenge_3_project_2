# main file to run the project pipeline:

from preprocessing.preprocessor import preprocessor_main
from feature_engineering.create_features import feature_engineering_main
from modelling.trees import trees_main
from preprocessing.scaling import scaling_main


def main() -> None:
    """
    This function runs the preprocessing, feature engineering and baseline modelling for trees.
    # Todo: Expand to all the other pipelines.
    :return: None
    """
    preprocessor_main(suppress_print=True)
    feature_engineering_main()
    scaling_main()
    trees_main()


if __name__ == '__main__':
    main()

