# auxiliary library to load data from the Excel file provided by professor Mitrovic.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# Global variables:
data_path: Path = Path('..', 'data', 'Project 2 Dataset.xls')


# Functions:
def load_and_save_data() -> pd.DataFrame:
    """
    Load the dataset from the Excel file.
    :return: pd.DataFrame: the dataframe containing the dataset.
    """
    # Load the dataset, ignore the first row:
    df: pd.DataFrame = pd.read_excel(data_path, sheet_name='Data', skiprows=1)

    return df
