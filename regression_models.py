import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Tuple, Dict, Any, List
import joblib

class RegressionSplitterTrainer:
    """
    Class for splitting data and training regression models. Includes model saving.
    """
    def __init__(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2, cv_folds: int = None, selected_models: List[str] = None):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.selected_models = selected_models or []  # If empty, train all
        self.models = self._get_models()

    def _get_models(self) -> Dict[str, Any]:
        """
        Get available regression models, filtered by selected_models if provided.
        """
        all_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor()
        }
        if self.selected_models:
            return {name: all_models[name] for name in self.selected_models if name in all_models}
        return all_models

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test.
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        if self.cv_folds:
            # For cross-validation, return full X, y
            return X, None, y, None
        else:
            return train_test_split(X, y, test_size=self.test_size, random_state=42)

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train selected regression models.
        """
        trained_models = {}
        for name, model in self.models.items():
            if self.cv_folds:
                scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
                model.fit(X_train, y_train)  # Fit on full for later use
                trained_models[name] = {'model': model, 'cv_scores': scores}
            else:
                model.fit(X_train, y_train)
                trained_models[name] = {'model': model}
        return trained_models

    def save_model(self, model: Any, model_name: str, path: str = 'models/'):
        """
        Save trained model using joblib.
        """
        joblib.dump(model, f"{path}{model_name}.joblib")    