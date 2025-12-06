"""Main entry point for product review sentiment analysis."""

import logging

from src.data_processing import preprocess_data
from src.base_model import BaselineModel
from src.utils import create_splits, save_results_md, load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the sentiment analysis pipeline."""
    # Load and preprocess
    df_reviews = load_data("data/amazon-cell-phones-reviews/reviews.csv")
    df_clean = preprocess_data(df_reviews, make_lowercase=True)

    # Split data
    splits = create_splits(df_clean)

    # Train baseline
    model = BaselineModel()
    model.train(splits.X_train, splits.y_train)

    # Evaluate
    metrics = model.evaluate(splits.X_test, splits.y_test)

    save_results_md(metrics, splits.stats)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()