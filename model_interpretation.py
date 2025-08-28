from typing import Dict, Any
import pandas as pd
import numpy as np

class ModelInterpreter:
    """
    Class for extracting feature importances or coefficients.
    """
    def __init__(self, models: Dict[str, Any]):
        self.models = models

    def get_feature_importances(self, feature_names: list) -> Dict[str, pd.DataFrame]:
        """
        Extract feature importances where applicable (tree-based models) or coefficients (linear models).
        """
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                importances[name] = imp_df
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients as "importance"
                coef = model.coef_
                if len(coef.shape) > 1:  # Multi-class or multi-output
                    coef = np.abs(coef).mean(axis=0)  # Average abs coef across classes
                else:
                    coef = np.abs(coef)
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coef
                }).sort_values(by='Coefficient', ascending=False)
                importances[name] = imp_df
        return importances