import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import dump, load
from mqt.predictor.ml.helper import create_feature_dict
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from qos.time_estimator.base_estimator import BaseEstimator
from qos.time_estimator.database import extract_jobs_from_ibm_quantum

logger = logging.getLogger(__name__)


class RegressionEstimator(BaseEstimator):
    def __init__(
        self,
        model: RegressorMixin | None = None,
        model_file: str | None = None,
        dataset_file: str | None = None,
    ):
        if dataset_file is not None:
            self.dataset_file = dataset_file
        else:
            self.dataset_file = (
                Path(__file__).absolute().parents[2]
                / "data/regression_dataset.npz"
            )
        if model is not None and model_file is not None:
            logger.warning("Both model and model file specified, using model")
            self.model = model
        elif model is not None:
            self.model = model
        elif model_file is not None:
            self.model_file = model_file
            try:
                self._load_model()
            except FileNotFoundError:
                logger.warning(
                    f"Model file {model_file} not found, training new model"
                )
                self._train_regression_model()
        else:
            self.model_file = (
                Path(__file__).absolute().parents[2]
                / "data/regression_model.joblib"
            )
            try:
                self._load_model()
            except FileNotFoundError:
                logger.info("Training new model")
                self._train_regression_model()

    def estimate_execution_time(
        self,
        circuits: list[QuantumCircuit],
        backend: Backend,
        **kwargs,
    ) -> float:
        """
        Estimate the execution time of a quantum job on a specified backend
        :param circuits: Circuits in the quantum job
        :param backend: Backend to be executed on
        :param kwargs: Additional arguments, like the run configuration
        :return: Estimated execution time
        """
        shots = min(kwargs.get("shots", 4000), backend.max_shots)
        features = self._extract_features(circuits, shots)
        return self.model.predict(features)[0]

    @staticmethod
    def _extract_features(
        circuits: list[QuantumCircuit], shots: int
    ) -> np.ndarray:
        """
        Extract features from a list of circuits
        :param circuits: Circuits to extract features from
        :param shots: Number of shots
        :return: Features
        """
        features = defaultdict(list)
        for circuit in circuits:
            circuit_features = create_feature_dict(circuit)
            for feature_name, feature_value in circuit_features.items():
                features[feature_name].append(feature_value)

        aggregate_features = {}
        for feature_name, feature_values in features.items():
            if feature_name in [
                "program_communication",
                "critical_depth",
                "entanglement_ratio",
                "parallelism",
                "liveness",
            ]:
                aggregate_features[feature_name] = np.array(
                    feature_values
                ).mean()
            else:
                aggregate_features[feature_name] = np.array(
                    feature_values
                ).sum()

        aggregate_features["shots"] = shots
        aggregate_features["circuit_count"] = len(circuits)

        return np.array(list(aggregate_features.values())).reshape(1, -1)

    @staticmethod
    def _tune_hyperparameters(
        model: RegressorMixin,
        parameter_grid: dict[str, list],
        x: np.ndarray,
        y: np.ndarray,
        scoring: str = "r2",
    ) -> tuple[RegressorMixin, float]:
        """
        Tune the hyperparameters of a model using grid search
        :param model: Model to be tuned
        :param parameter_grid: Parameter grid for grid search
        :param x: Independent variables
        :param y: Dependent variables
        :param scoring: Scoring method
        :return: Tuned model and score
        """
        # K-fold cross-validation
        # cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        # Grid search
        grid_search = GridSearchCV(
            model, parameter_grid, scoring=scoring, n_jobs=-1, cv=cv
        )
        # Tune hyperparameters
        grid_search.fit(x, y)

        # Get best model based on scoring method
        return grid_search.best_estimator_, grid_search.best_score_

    def _choose_best_model(
        self,
        models: dict[str, RegressorMixin],
        parameter_grids: dict[str, dict[str, list]],
        x: np.ndarray,
        y: np.ndarray,
    ):
        """
        Choose the best model based on the scoring method
        :param models: Models to be evaluated
        :param parameter_grids: Parameter grids for grid search
        :param x: Independent variables
        :param y: Dependent variables
        :return: Best model
        """
        best_model = None
        best_score = 0
        best_model_name = None
        # Evaluate models
        for model_name, model in models.items():
            # Get score for the tuned model
            logger.info(f"Evaluating {model_name}")
            tuned_model, score = self._tune_hyperparameters(
                model, parameter_grids[model_name], x, y
            )
            logger.info(f"{model_name} score: {score}")
            if score > best_score:
                best_score = score
                best_model = tuned_model
                best_model_name = model_name

        logger.info(f"Best model: {best_model_name} with score {best_score}")

        self.model = best_model

    def _save_model(self) -> None:
        """
        Save the model to a file
        """
        dump(self.model, self.model_file)
        logger.info(f"Model saved to {self.model_file}")

    def _load_model(self) -> None:
        """
        Load the model from a file
        """
        self.model = load(self.model_file)

    @staticmethod
    def _get_available_models() -> dict[str, RegressorMixin]:
        """
        Get the models to be evaluated
        :return: Models to be evaluated
        """
        models = {
            "Extra Trees": ExtraTreesRegressor(n_jobs=-1, random_state=0),
            "Random Forest": RandomForestRegressor(n_jobs=-1, random_state=0),
            "Gradient Boosting": GradientBoostingRegressor(random_state=0),
            "AdaBoost": AdaBoostRegressor(random_state=0),
            "Histogram Gradient Boosting": HistGradientBoostingRegressor(
                random_state=0
            ),
            "Polynomial Regression": make_pipeline(
                PolynomialFeatures(degree=3, include_bias=False),
                LinearRegression(),
            ),
        }

        return models

    @staticmethod
    def _get_model_parameter_grids() -> dict[str, dict[str, list]]:
        """
        Get the parameter grids for grid search
        :return: Parameter grids for grid search
        """
        parameter_grids = {
            "Extra Trees": {
                "n_estimators": [100, 300, 500, 700],
                "max_depth": [None, 5, 10, 30, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10, 20, 30],
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": [None, 10, 30, 50, 70],
                "bootstrap": [True, False],
            },
            "Random Forest": {
                "n_estimators": [100, 300, 500, 700],
                "max_depth": [None, 5, 10, 30, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10, 20, 30],
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": [None, 10, 30, 50, 70],
                "bootstrap": [True, False],
            },
            "Gradient Boosting": {
                "learning_rate": [0.001, 0.01, 0.1, 1],
                "n_estimators": [100, 300, 500, 700],
                "subsample": [0.3, 0.5, 0.7, 1.0],
                "max_depth": [None, 5, 10, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10, 20, 30],
                "max_features": [None, "sqrt", "log2"],
                "max_leaf_nodes": [None, 10, 30, 50, 70],
            },
            "AdaBoost": {
                "n_estimators": [100, 300, 500, 700],
                "learning_rate": [0.001, 0.01, 0.1, 1],
                "loss": ["linear", "square", "exponential"],
            },
            "Histogram Gradient Boosting": {
                "learning_rate": [0.001, 0.01, 0.1, 1],
                "max_iter": [100, 300, 500, 700],
                "max_leaf_nodes": [10, 30, 50, 70],
                "max_depth": [None, 5, 10, 30, 50],
                "min_samples_leaf": [5, 10, 20, 30],
                "l2_regularization": [0, 0.01, 0.1, 1],
            },
            "Polynomial Regression": {
                "polynomialfeatures__degree": [2, 3, 4],
            },
        }

        return parameter_grids

    def _create_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the dataset
        """
        x = []
        y = []

        for job in extract_jobs_from_ibm_quantum():
            try:
                logger.info("Extracting features from job %s", job.job_id())
                x.append(
                    self._extract_features(
                        job.circuits(), job.backend_options()["shots"]
                    )
                )
                y.append(job.result().time_taken)
            except Exception as e:
                logger.warning(
                    "Failed to extract features from job %s: %s",
                    job.job_id(),
                    e,
                )

        x = np.concatenate(x, axis=0)
        y = np.array(y)

        np.savez(self.dataset_file, x=x, y=y)

        return x, y

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the dataset
        :return: Dataset with independent and dependent variables
        """
        try:
            dataset = np.load(self.dataset_file)
            x = dataset["x"]
            y = dataset["y"]
        except FileNotFoundError:
            logger.warning("Dataset not found, creating new dataset")
            x, y = self._create_dataset()

        return x, y

    def _train_regression_model(self) -> None:
        """
        Train a regression model
        """
        x, y = self._load_dataset()
        models = self._get_available_models()
        parameter_grids = self._get_model_parameter_grids()
        self._choose_best_model(
            models,
            parameter_grids,
            x,
            y,
        )
        self._save_model()