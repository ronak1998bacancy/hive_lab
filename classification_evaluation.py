from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
import pandas as pd

class ClassificationEvaluator:
    """
    Class for evaluating classification models.
    """
    def evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Compute classification evaluation metrics.
        """
        evaluations = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            evaluations[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted')
            }
        return evaluations