"""
This is a boilerplate pipeline 'make_submission'
generated using Kedro 0.17.6
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def train_model(X: np.ndarray, y: np.ndarray, iterations: int, cat_features: List, random_state: int,
                parameters: Dict) -> CatBoostClassifier:
    # Define model
    model = CatBoostClassifier(
        random_seed=random_state,
        **parameters,
        iterations=iterations,
    )

    # Train model on
    model.fit(X, y, cat_features=cat_features)

    return model


def make_submission(model: CatBoostClassifier, X_test: np.ndarray) -> np.ndarray:
    preds_proba = model.predict_proba(X_test)

    submission = pd.DataFrame({
        "SK_ID_CURR": X_test.SK_ID_CURR,
        "TARGET": np.array(preds_proba)[:, 1],
    })

    return submission