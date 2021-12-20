import numpy as np
import pandas as pd
from typing import List
import gc

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


def agg_categorical(df: pd.DataFrame, parent_var: str, df_name: str) -> pd.DataFrame:
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.

    Parameters
    --------
    df : dataframe
        The dataframe to calculate the value counts for.

    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.

    """

    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:, idx]

    return categorical


def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level
    at the client level

    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])

    Returns:
        df_client (dataframe): aggregated numeric stats at the client level.
        Each client will have a single row with all the numeric data aggregated
    """

    # Aggregate the numeric columns
    df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])

    # If there are categorical variables
    if any(df.dtypes == 'category'):

        # Count the categorical columns
        df_counts = agg_categorical(df, parent_var=group_vars[0], df_name=df_names[0])

        # Merge the numeric and categorical
        df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how='outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        # Merge to get the client id in dataframe
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])


    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        gc.enable()
        del df_agg
        gc.collect()

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])

    # Memory management
    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client


def merge_with_main_datasets(train: pd.DataFrame, test: pd.DataFrame, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train = train.merge(df, on='SK_ID_CURR', how='left')
    test = test.merge(df, on='SK_ID_CURR', how='left')

    return train, test