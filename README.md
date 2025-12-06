# Product Review Intelligence

Modern sentiment analysis pipeline for Amazon product reviews, demonstrating evolution from classical ML to state-of-the-art transformer models.

## Project Objectives

- Modernize legacy sentiment analysis approaches with 2024/2025 techniques
- Demonstrate full ML pipeline: data preprocessing → model training → evaluation → deployment
- Build portfolio showcasing NLP engineering and research capabilities
- Establish baseline for transformer fine-tuning and RAG implementation

## Project Structure
```
product-review-intelligence/
├── data/
│   └── amazon-cell-phones-reviews/
│       ├── reviews.csv           # Main dataset (download required)
│       └── items.csv              # Product metadata (optional)
├── notebooks/
│   └── sentiment_analysis.ipynb   # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── base_model.py              # Baseline TF-IDF + LogReg model
│   └── utils.py                   # Metrics, splits, persistence
├── main.py                        # Pipeline orchestrator
├── results.md                     # Model evaluation metrics
├── pyproject.toml                 # Dependencies (UV/pip)
└── README.md
```

## Technical Approach

### Phase 2: Baseline Implementation (Completed)

**Data Preprocessing:**
- Deduplication by product-review pair (67,986 → 59,605 reviews)
- Binary sentiment labels (1-2 stars = negative, 4-5 stars = positive)
- Removed neutral reviews (3 stars) for clarity
- Text cleaning: HTML/URL removal, whitespace normalization
- 60/20/20 train/validation/test split

**Model Architecture:**
- TF-IDF vectorization (5,000 features, bigrams)
- Logistic Regression with balanced class weights
- Handles class imbalance (72% positive, 28% negative)

**Results:** 92.02% accuracy, 94.34% weighted F1-score. See `results.md` for detailed metrics.

### Planned Enhancements

- **Phase 3:** Fine-tune Llama-3/Mistral with LoRA
- **Phase 4:** RAG-based review Q&A system
- **Phase 5:** Gradio deployment to HuggingFace Spaces
- **Phase 6:** Documentation and portfolio polish

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Alternatively: brew install uv (macOS)

# Clone repository
git clone <your-repo-url>
cd product-review-intelligence

# Setup environment
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

## Dataset Setup

Download the [Amazon Cell Phone Reviews dataset](https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews) from Kaggle.

Place files in:
```
data/amazon-cell-phones-reviews/
├── reviews.csv
└── items.csv
```

## Usage
```bash
python main.py
```

**Output:** Trained model evaluation and `results.md` metrics report.

## Results

See `results.md` for comprehensive baseline model performance metrics.