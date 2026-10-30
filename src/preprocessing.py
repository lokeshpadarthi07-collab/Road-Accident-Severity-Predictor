"""
Preprocessing Pipeline Module for Road Accident Severity Predictor.

Constructs scikit-learn ColumnTransformer for categorical and numerical features,
manages target encoding, handles missing values, and ensures consistent transformations.
"""

import os
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def build_preprocessor(
    categorical_features: List[str],
    numerical_features: List[str]
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer with robust missing value imputation,
    scaling, and one-hot encoding with handle_unknown='ignore'.
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="passthrough"
    )

    return preprocessor


class TargetEncoder:
    """
    Custom Target Encoder maintaining consistent class order (e.g. Slight=0, Serious=1, Fatal=2).
    """

    def __init__(self, class_order: Optional[List[str]] = None):
        self.class_order = class_order or ["Slight", "Serious", "Fatal"]
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes_: np.ndarray = np.array([])

    def fit(self, y: pd.Series):
        unique_classes = sorted(list(y.dropna().unique()))
        # Prioritize specified class_order if all elements are present
        if set(self.class_order) == set(unique_classes):
            self.classes_ = np.array(self.class_order)
        else:
            self.classes_ = np.array(unique_classes)

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y: pd.Series) -> np.ndarray:
        return np.array([self.class_to_idx[val] for val in y])

    def fit_transform(self, y: pd.Series) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_indices: np.ndarray) -> np.ndarray:
        return np.array([self.idx_to_class[int(idx)] for idx in y_indices])


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as e:
        logger.warning(f"Could not extract feature names via get_feature_names_out: {e}")
        return []


def save_preprocessor_artifacts(
    preprocessor: ColumnTransformer,
    target_encoder: TargetEncoder,
    preprocessor_path: str = "models/preprocessor.joblib",
    encoder_path: str = "models/label_encoder.joblib"
) -> None:
    """Save preprocessor and target encoder to disk."""
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(target_encoder, encoder_path)
    logger.info(f"Saved preprocessor to: {preprocessor_path}")
    logger.info(f"Saved target encoder to: {encoder_path}")


def load_preprocessor_artifacts(
    preprocessor_path: str = "models/preprocessor.joblib",
    encoder_path: str = "models/label_encoder.joblib"
) -> Tuple[ColumnTransformer, TargetEncoder]:
    """Load preprocessor and target encoder from disk."""
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Target encoder not found at: {encoder_path}")

    preprocessor = joblib.load(preprocessor_path)
    target_encoder = joblib.load(encoder_path)
    return preprocessor, target_encoder
