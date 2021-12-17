import logging
import pandas as pd
import numpy as np
from typing import List, Dict

# Models
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


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


def drop_target(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    y = data["TARGET"]
    X = data.drop("TARGET", axis=1)

    return [X, y]


def split_data(X: pd.DataFrame, y: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.
        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.
        Returns:
            A list containing split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]


def validate_model(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, cat_features: List, random_state: int, parameters: Dict) -> CatBoostClassifier:
    # train the model
    model = CatBoostClassifier(
        loss_function=parameters["loss_function"],
        eval_metric=parameters["eval_metric"],
        use_best_model=parameters["use_best_model"],
        random_seed=random_state,
        early_stopping_rounds=parameters["early_stopping_rounds"],
    )

    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_valid, y_valid))

    logger = logging.getLogger(__name__)
    logger.info('Best Iteration: {}'.format(model.best_iteration_))
    logger.info('Best Score: {}'.format(model.best_score_['validation']['AUC']))

    return model


