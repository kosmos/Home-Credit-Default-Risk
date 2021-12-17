import numpy as np
import pandas as pd


"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.17.6
"""


def clean_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """"
    A function that clean field DAYS_EMPLOYED from anomaly value 365243
    """
    df_ = df.copy()

    # Create an anomalous flag column
    df_['DAYS_EMPLOYED_ANOM'] = df_["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    df_['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    return df_