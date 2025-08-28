from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
import numpy as np
import pandas as pd

class RegressionEvaluator:
    """
    Class for evaluating regression models.
    """
    def evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Compute regression evaluation metrics.
        """
        evaluations = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            evaluations[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            }
        return evaluations