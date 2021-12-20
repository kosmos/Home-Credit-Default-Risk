import logging
import pandas as pd
import numpy as np
from typing import List, Dict

# Models
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


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


def validate_model(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, cat_features: List, random_state: int, parameters: Dict) -> int:
    # Define model
    model = CatBoostClassifier(
        random_seed=random_state,
        **parameters,
        use_best_model=True,
    )

    # Train model on
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_valid, y_valid))

    # Log results
    logger = logging.getLogger(__name__)
    logger.info('Best Iteration: {}'.format(model.best_iteration_))
    logger.info('Best Score: {}'.format(model.best_score_['validation']['AUC']))

    return model.best_iteration_
