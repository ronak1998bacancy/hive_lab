import pandas as pd
from typing import Dict, List, Any
import joblib
import json
import os
from data_collection import DataCollector
from data_cleaning import DataCleaner
from eda_computation import EDAComputer
from feature_engineering import FeatureEngineer
from model_interpretation import ModelInterpreter

class MLPipelineOrchestrator:
    """
    Orchestrates the full ML pipeline.
    """
    def __init__(self, file_path: str, target_column: str, task_type: str = 'regression',
                 drop_columns: List[str] = None, encoding_methods: Dict[str, str] = None,
                 param_grids: Dict[str, Dict] = None, test_size: float = 0.2, cv_folds: int = None,
                 selected_models: List[str] = None):
        self.file_path = file_path
        self.target_column = target_column
        self.task_type = task_type
        self.drop_columns = drop_columns or []
        self.encoding_methods = encoding_methods or {}
        self.param_grids = param_grids or {}
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.selected_models = selected_models or []  # List of model names to train
        self.data = None
        self.eda_stats = None
        self.trained_models = None
        self.tuned_models = None
        self.evaluations = None
        self.importances = None
        self.predictions = None
        self.preprocessor = None
        self.metadata = None

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the full pipeline and return results.
        """
        # Create models directory if not exists
        os.makedirs('models', exist_ok=True)

        # Step 1: Data Collection
        collector = DataCollector(self.file_path)
        self.data = collector.load_data()

        # Step 2: Data Cleaning
        cleaner = DataCleaner(self.data, self.target_column, self.task_type)
        self.data = cleaner.clean_data(self.drop_columns)
        self.target_column = cleaner.target_column

        # Save metadata after cleaning (before engineering)
        self.features = list(self.data.columns.drop(self.target_column))
        categorical_cols = self.data[self.features].select_dtypes(include=['object', 'category']).columns.tolist()
        self.metadata = {
            'features': self.features,
            'categorical_cols': categorical_cols,
            'categories': {col: list(self.data[col].unique()) for col in categorical_cols},
            'target_column': self.target_column,
            'task_type': self.task_type
        }
        with open('models/metadata.json', 'w') as f:
            json.dump(self.metadata, f)

        # Step 3: EDA Computation
        eda = EDAComputer(self.data)
        self.eda_stats = eda.compute_eda()

        # Step 4: Feature Engineering
        engineer = FeatureEngineer(self.data, self.target_column, self.encoding_methods)
        self.data = engineer.engineer_features()
        self.preprocessor = engineer.get_preprocessor()
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')

        # Step 5: Dynamically import task-specific modules
        if self.task_type == 'regression':
            from regression_models import RegressionSplitterTrainer
            from regression_tuning import RegressionTuner
            from regression_evaluation import RegressionEvaluator
            SplitterTrainer = RegressionSplitterTrainer
            Tuner = RegressionTuner
            Evaluator = RegressionEvaluator
        elif self.task_type == 'classification':
            from classification_models import ClassificationSplitterTrainer
            from classification_tuning import ClassificationTuner
            from classification_evaluation import ClassificationEvaluator
            SplitterTrainer = ClassificationSplitterTrainer
            Tuner = ClassificationTuner
            Evaluator = ClassificationEvaluator
        else:
            raise ValueError("Unsupported task_type. Use 'regression' or 'classification'.")

        # Step 6: Data Splitting and Model Training (with selected models)
        splitter_trainer = SplitterTrainer(self.data, self.target_column, self.test_size, self.cv_folds, selected_models=self.selected_models)
        X_train, X_test, y_train, y_test = splitter_trainer.split_data()
        self.trained_models = splitter_trainer.train_models(X_train, y_train)

        # Step 7: Hyperparameter Tuning
        tuner = Tuner(self.trained_models, self.param_grids)
        self.tuned_models = tuner.tune_models(X_train, y_train)

        # Step 8: Model Evaluation
        if not self.cv_folds:  # Evaluate on test set if no CV
            evaluator = Evaluator()
            self.evaluations = evaluator.evaluate_models(self.tuned_models, X_test, y_test)
            # Collect predictions for plotting
            self.predictions = {'y_test': y_test}
            for name, model in self.tuned_models.items():
                y_pred = model.predict(X_test)
                self.predictions[name] = y_pred

        # Step 9: Model Interpretation
        interpreter = ModelInterpreter(self.tuned_models)
        self.importances = interpreter.get_feature_importances(X_train.columns.tolist())

        # Step 10: Save models
        if self.tuned_models:
            for name, model in self.tuned_models.items():
                splitter_trainer.save_model(model, name)

        return {
            'eda_stats': self.eda_stats,
            'evaluations': self.evaluations,
            'importances': self.importances,
            'models': self.tuned_models,
            'predictions': self.predictions
        }