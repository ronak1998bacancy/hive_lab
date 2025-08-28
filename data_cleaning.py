import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    """
    Class for cleaning data: fixing column names, dropping duplicates/invalids, ensuring numeric target for regression.
    """
    def __init__(self, data: pd.DataFrame, target_column: str, task_type: str = 'regression'):
        self.data = data
        self.target_column = target_column
        self.task_type = task_type  # 'regression' or 'classification'

    def clean_data(self, drop_columns: List[str] = None) -> pd.DataFrame:
        """
        Perform data cleaning steps.
        """
        try:
            # Store original target column name for reference
            original_target = self.target_column

            # Fix column names: strip whitespace, lowercase, replace spaces with underscores
            self.data.columns = self.data.columns.str.strip().str.lower().str.replace(' ', '_')

            # Update target column name to match the cleaned format
            self.target_column = self.target_column.strip().lower().replace(' ', '_')

            # Check if target column exists after cleaning
            if self.target_column not in self.data.columns:
                available_columns = list(self.data.columns)
                raise KeyError(f"Target column '{original_target}' (cleaned as '{self.target_column}') not found in dataset. Available columns: {available_columns}")

            # Drop specified columns if any
            if drop_columns:
                # Clean the drop_columns names to match the cleaned column format
                cleaned_drop_columns = [col.strip().lower().replace(' ', '_') for col in drop_columns]
                # Ensure target is not in drop_columns
                if self.target_column in cleaned_drop_columns:
                    raise ValueError(f"Cannot drop the target column '{self.target_column}'. Please remove it from drop_columns.")
                self.data = self.data.drop(columns=cleaned_drop_columns, errors='ignore')

            # Re-check if target still exists after drops (in case of overlap or errors)
            if self.target_column not in self.data.columns:
                raise ValueError(f"Target column '{self.target_column}' is missing after dropping columns. Ensure it is not included in drop_columns.")

            # Drop duplicates
            self.data = self.data.drop_duplicates()

            # Drop rows with invalid (NaN) target
            self.data = self.data.dropna(subset=[self.target_column])

            # Ensure target is suitable for the task
            n_unique = self.data[self.target_column].nunique()
            if self.task_type == 'regression':
                if not pd.api.types.is_numeric_dtype(self.data[self.target_column]):
                    raise ValueError(f"Target column '{self.target_column}' must be numeric for regression.")
                if n_unique < 2:
                    raise ValueError(f"Target column '{self.target_column}' has fewer than 2 unique values, unsuitable for regression.")
            elif self.task_type == 'classification':
                if n_unique > 50:  # Arbitrary threshold to prevent fitting on continuous targets
                    raise ValueError(f"Target column '{self.target_column}' has too many unique values ({n_unique}) for classification; consider regression or binning the values.")
                if n_unique < 2:
                    raise ValueError(f"Target column '{self.target_column}' has fewer than 2 unique values, unsuitable for classification.")
                # Convert to category and label encode to ensure integer labels for classifiers
                self.data[self.target_column] = self.data[self.target_column].astype('category')
                encoder = LabelEncoder()
                self.data[self.target_column] = encoder.fit_transform(self.data[self.target_column])

            return self.data

        except KeyError as e:
            raise KeyError(f"Column error during data cleaning: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during data cleaning: {str(e)}")