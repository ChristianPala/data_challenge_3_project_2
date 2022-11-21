# library to feature engineering some features from the project 2 dataset in the Data Challenge III course,
# SUPSI Bachelor in Data Science, 2022.

# Libraries:
import pandas as pd
import numpy as np

# Data manipulation:
from pathlib import Path


# Global variables:
# Path to the dataset:
data_path: Path = Path('..', 'data')
excel_file: Path = Path(data_path, 'Project 2 Dataset.xls')

