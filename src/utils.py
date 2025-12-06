"""Utility functions for data loading, splitting, metrics, and result persistence."""

import logging
from typing import Dict, Tuple, NamedTuple

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class SplitData(NamedTuple):
    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    stats: dict

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def create_splits(df: pd.DataFrame) -> SplitData:
    """
    Splits the data into train, validation, and test sets with 60/20/20 ratio
    """
    X = df['clean_text']
    y = df['sentiment']

    # 60/40 Train and Rest split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    # 20/20 Validation and Test split from Rest
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    stats = {
        'total': len(df),
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test),
        'neg_count': (y == 0).sum(),
        'pos_count': (y == 1).sum()
    }

    logger.info(f"Train: {X_train.shape[0]} samples")
    logger.info(f"Val:   {X_val.shape[0]} samples")
    logger.info(f"Test:  {X_test.shape[0]} samples")

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, stats)

def extract_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float | str]:
    """Extract classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary with accuracy, weighted F1, and classification report.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_neg': report['0']['precision'],
        'recall_neg': report['0']['recall'],
        'precision_pos': report['1']['precision'],
        'recall_pos': report['1']['recall'],
        'report_str': classification_report(y_true, y_pred)
    }

def save_results_md(metrics, dataset_stats, filepath="results.md"):
    """Persist metrics to markdown."""
    content = f"""# Baseline Results

    ## Dataset
    - Train: {dataset_stats['train']}
    - Val: {dataset_stats['val']}
    - Test: {dataset_stats['test']}

    ## Metrics
    - Accuracy: {metrics['accuracy']:.2%}
    - F1 (weighted): {metrics['f1_weighted']:.2%}
    ```
    {metrics['report_str']}
    ```
    """

    with open(filepath, 'w') as f:
        f.write(content)