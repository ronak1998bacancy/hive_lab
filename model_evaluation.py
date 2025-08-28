from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
import numpy as np
import pandas as pd

class ModelEvaluator:
    """
    Class for evaluating models.
    """
    def __init__(self, task_type: str = 'regression'):
        self.task_type = task_type

    def evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics.
        """
        evaluations = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            if self.task_type == 'regression':
                evaluations[name] = {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred)
                }
            elif self.task_type == 'classification':
                evaluations[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1': f1_score(y_test, y_pred, average='weighted')
                }
        return evaluations