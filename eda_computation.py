import pandas as pd
from typing import Dict

class EDAComputer:
    """
    Class for computing Exploratory Data Analysis (EDA) statistics.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def compute_eda(self) -> Dict:
        """
        Generate EDA stats: min/max for numerics, unique samples for categoricals.
        """
        eda_stats = {}

        # Numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            eda_stats[col] = {
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'mean': self.data[col].mean(),
                'std': self.data[col].std()
            }

        # Categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            uniques = self.data[col].unique()[:10]  # Sample up to 10 uniques
            eda_stats[col] = {
                'unique_samples': list(uniques),
                'n_unique': self.data[col].nunique()
            }

        return eda_stats