import numpy as np
import pandas as pd
from typing import List

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

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


def transform_categorical_features(df: pd.DataFrame) -> (pd.DataFrame, List):
    """
    Transform categorical features for CatBoost algorithm

    Args:
        df: dataframe to transform

    Returns:

    """
    d = df.copy()
    cat_features = d.columns[np.where(d.dtypes != float)[0]].values.tolist()
    d[cat_features] = d[cat_features].astype(str)

    return d, cat_features


def drop_target(data: pd.DataFrame, target_column: str) -> (pd.DataFrame, pd.DataFrame):
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    return [X, y]


def fe_poly_features(train: pd.DataFrame, test: pd.DataFrame, poly_features_columns: List) \
        -> (pd.DataFrame,  pd.DataFrame):
    # Make a new dataframe for polynomial features
    poly_features = train[poly_features_columns]
    poly_features_test = test[poly_features_columns]

    imputer = SimpleImputer(strategy="median")

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)

    # Create a dataframe of the features
    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                             'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                  'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = train['SK_ID_CURR']
    train_poly = train.merge(poly_features, on='SK_ID_CURR', how='left')

    # Merge polnomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
    test_poly = test.merge(poly_features_test, on='SK_ID_CURR', how='left')

    # Align the dataframes
    train_poly, test_poly = train_poly.align(test_poly, join='inner', axis=1)

    return train_poly, test_poly


def fe_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    df_['CREDIT_INCOME_PERCENT'] = df_['AMT_CREDIT'] / df_['AMT_INCOME_TOTAL']
    df_['ANNUITY_INCOME_PERCENT'] = df_['AMT_ANNUITY'] / df_['AMT_INCOME_TOTAL']
    df_['CREDIT_TERM'] = df_['AMT_ANNUITY'] / df_['AMT_CREDIT']
    df_['INCOME_PER_PERSON'] = df_['AMT_INCOME_TOTAL'] / df_['CNT_FAM_MEMBERS']
    df_['INCOME_CREDIT_PERC'] = df_['AMT_INCOME_TOTAL'] / df_['AMT_CREDIT']

    # TODO: To fix
    # df_['DAYS_EMPLOYED_PERCENT'] = df_['DAYS_EMPLOYED'] / df_['DAYS_BIRTH']

    return df_


def convert_types(df, print_info=True):
    original_memory = df.memory_usage().sum()

    # Iterate through each column
    for c in df:

        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)

        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')

        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)

        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)

        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)

    new_memory = df.memory_usage().sum()

    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')

    return df


def agg_numeric(df: pd.DataFrame, parent_var: str, df_name: str) -> pd.DataFrame:
    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns

    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis=1, return_index=True)
    agg = agg.iloc[:, idx]

    return agg


def merge_with_main_datasets(train: pd.DataFrame, test: pd.DataFrame, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train = train.merge(df, on='SK_ID_CURR', how='left')
    test = test.merge(df, on='SK_ID_CURR', how='left')

    return train, test