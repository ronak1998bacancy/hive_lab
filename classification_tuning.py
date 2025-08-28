from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
import pandas as pd

class ClassificationTuner:
    """
    Class for hyperparameter tuning classification models using GridSearchCV.
    """
    def __init__(self, models: Dict[str, Dict[str, Any]], param_grids: Dict[str, Dict]):
        self.models = models  # From trainer: {'name': {'model': model}}
        self.param_grids = param_grids  # e.g., {'RandomForest': {'n_estimators': [50, 100], ...}}

    def tune_models(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for classification.
        """
        tuned_models = {}
        for name, info in self.models.items():
            model = info['model']
            if name in self.param_grids:
                grid = GridSearchCV(model, self.param_grids[name], cv=cv, scoring='f1_weighted')
                grid.fit(X_train, y_train)
                tuned_models[name] = grid.best_estimator_
            else:
                tuned_models[name] = model  # No tuning if no grid
        return tuned_models