"""Text preprocessing and data cleaning for review sentiment analysis."""

import logging
import re

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def remove_html_and_urls(text: str) -> str:
    """Remove HTML tags and URLs from text.
    
    Args:
        text: Raw text potentially containing HTML/URLs.
        
    Returns:
        Cleaned text with HTML and URLs removed.
    """
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)
    return " ".join(text.split())


def clean_text(text: str, lowercase: bool = False) -> str:
    """Clean text with optional lowercasing.
    
    Args:
        text: Raw review text.
        lowercase: If True, convert to lowercase (for classical ML).
                   Set False for transformer models.
                   
    Returns:
        Cleaned text string.
    """
    text = remove_html_and_urls(text)
    return text.lower() if lowercase else text


def preprocess_data(
    df: pd.DataFrame,
    make_lowercase: bool = False,
    min_text_length: int = 10,
) -> pd.DataFrame:
    """Preprocess review dataset for sentiment analysis.
    
    Performs:
        - Drops rows with missing body/rating
        - Removes duplicate reviews per product
        - Filters short reviews
        - Creates binary sentiment labels (drops neutral 3-star reviews)
        - Cleans text
    
    Args:
        df: Raw DataFrame with 'body', 'rating', 'asin' columns.
        make_lowercase: Whether to lowercase text.
        min_text_length: Minimum review length to keep.
        
    Returns:
        Preprocessed DataFrame with 'clean_text' and 'sentiment' columns.
    """
    initial_count = len(df)
    
    df = df.dropna(subset=["body", "rating"])
    df = df.drop_duplicates(subset=["asin", "body"], keep="first")
    df = df[df["body"].str.len() >= min_text_length]

    # Binary sentiment: 1-2 stars = negative (0), 4-5 stars = positive (1)
    df = df[df["rating"] != 3]
    df["sentiment"] = (df["rating"] >= 4).astype(int)

    df["clean_text"] = df["body"].apply(
        lambda x: clean_text(x, lowercase=make_lowercase)
    )
    
    logger.info(f"Preprocessed {len(df)}/{initial_count} reviews")
    return df
