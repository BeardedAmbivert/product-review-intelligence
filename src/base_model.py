"""Baseline TF-IDF + Logistic Regression model for sentiment classification."""

import logging
from typing import Dict

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils import extract_metrics

logger = logging.getLogger(__name__)


class BaselineModel:
    """TF-IDF + Logistic Regression baseline for sentiment analysis.
    
    Attributes:
        vectorizer: Fitted TF-IDF vectorizer.
        model: Trained logistic regression classifier.
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        max_iter: int = 1000,
    ) -> None:
        """Initialize the baseline model.
        
        Args:
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: Range of n-grams to extract.
            max_iter: Max iterations for logistic regression.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
        )

    def _vectorize(self, data: pd.Series, fit: bool = False) -> spmatrix:
        """Transform text to TF-IDF features.
        
        Args:
            data: Text series to vectorize.
            fit: If True, fit the vectorizer (training). Otherwise transform only.
            
        Returns:
            Sparse TF-IDF feature matrix.
        """
        if fit:
            return self.vectorizer.fit_transform(data)
        return self.vectorizer.transform(data)

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Train the model on training data.
        
        Args:
            X_train: Training text data.
            y_train: Training labels.
        """
        logger.info("Vectorizing training data...")
        X_train_vec = self._vectorize(X_train, fit=True)
        
        logger.info("Training logistic regression...")
        self.model.fit(X_train_vec, y_train)
        logger.info("Training complete.")

    def predict(self, X: pd.Series) -> pd.Series:
        """Generate predictions for input text.
        
        Args:
            X: Text data to classify.
            
        Returns:
            Predicted labels.
        """
        X_vec = self._vectorize(X, fit=False)
        return self.model.predict(X_vec)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, float | str]:
        """Evaluate model on test data.
        
        Args:
            X_test: Test text data.
            y_test: Ground truth labels.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.predict(X_test)
        metrics = extract_metrics(y_test, y_pred)
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics

