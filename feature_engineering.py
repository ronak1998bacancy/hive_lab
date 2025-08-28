import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Dict

def log_transform(X):
    return np.log1p(np.maximum(X, 0))

def clip_outliers(X):
    Q1 = np.quantile(X, 0.25, axis=0)
    Q3 = np.quantile(X, 0.75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(X, lower, upper)

class FeatureEngineer:
    """
    Class for feature engineering: handling missing, skew, outliers, collinearity, encoding, scaling.
    Uses ColumnTransformer for savable preprocessor.
    """
    def __init__(self, data: pd.DataFrame, target_column: str, encoding_methods: Dict[str, str] = None):
        self.data = data
        self.target_column = target_column
        self.encoding_methods = encoding_methods or {}  # e.g., {'col1': 'onehot', 'col2': 'label'}
        self.preprocessor = None

    def engineer_features(self, missing_threshold: float = 0.5, variance_threshold: float = 0.0) -> pd.DataFrame:
        """
        Perform feature engineering steps and build savable preprocessor.
        """
        # Drop high-missing columns
        missing_ratios = self.data.isnull().mean()
        high_missing = missing_ratios[missing_ratios > missing_threshold].index
        self.data = self.data.drop(columns=high_missing)

        # Drop zero-variance columns
        low_variance = self.data.columns[self.data.nunique() <= 1]
        self.data = self.data.drop(columns=low_variance)

        # Handle collinearity: drop one of highly correlated pairs (>0.9)
        numeric_cols = self.data.select_dtypes(include=['number']).columns.drop(self.target_column, errors='ignore')
        corr_matrix = self.data[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        self.data = self.data.drop(columns=to_drop)

        # Updated column lists after drops
        numeric_cols = self.data.select_dtypes(include=['number']).columns.drop(self.target_column, errors='ignore')
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        # Transformers using global functions
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('log', FunctionTransformer(func=log_transform, kw_args=None, feature_names_out='one-to-one')),
            ('outlier', FunctionTransformer(func=clip_outliers, kw_args=None, feature_names_out='one-to-one')),
            ('scaler', StandardScaler())
        ])

        # For categoricals: separate by encoding method
        onehot_cols = [col for col in categorical_cols if self.encoding_methods.get(col) == 'onehot']
        label_cols = [col for col in categorical_cols if self.encoding_methods.get(col) == 'label']
        auto_cols = list(set(categorical_cols) - set(onehot_cols) - set(label_cols) - {self.target_column})

        onehot_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
        ])

        label_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('label', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # For auto_cols: low card onehot, high label
        auto_onehot = [col for col in auto_cols if self.data[col].nunique() <= 10]
        auto_label = [col for col in auto_cols if self.data[col].nunique() > 10]

        # Build ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipe, numeric_cols),
                ('onehot_user', onehot_pipe, onehot_cols),
                ('label_user', label_pipe, label_cols),
                ('onehot_auto', onehot_pipe, auto_onehot),
                ('label_auto', label_pipe, auto_label)
            ],
            remainder='passthrough'  # For other cols
        )

        # Fit and transform (exclude target)
        X = self.data.drop(columns=[self.target_column])
        X_transformed = self.preprocessor.fit_transform(X)
        # Get new column names
        new_cols = self.preprocessor.get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=new_cols, index=self.data.index)
        self.data = pd.concat([X_transformed_df, self.data[[self.target_column]]], axis=1)

        return self.data

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Return the fitted preprocessor for saving.
        """
        return self.preprocessor