import pandas as pd
from typing import Union

class DataCollector:
    """
    Class for loading data from CSV or Excel files with error handling.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the provided file path.
        Handles CSV and Excel formats with error resilience.
        """
        try:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Only CSV and Excel are supported.")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")