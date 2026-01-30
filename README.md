# UFC Fight Outcome Prediction System

A machine learning system that predicts UFC fight outcomes (Win/Loss) using historical fighter data and engineered features.

## Overview

This project implements an end-to-end ML pipeline for predicting UFC fight results:
1. **Data Collection**: Scrapes fighter statistics and fight history from ufcstats.com
2. **Preprocessing**: Cleans and normalizes raw data into structured datasets
3. **Feature Engineering**: Creates comprehensive features (54 total) including fighter stats, recent form, streaks, strength of schedule, and head-to-head comparisons
4. **Model Training**: Trains multiple ML models (Random Forest, XGBoost, LightGBM, etc.) with hyperparameter tuning
5. **Evaluation**: Evaluates models with metrics and visualizations (confusion matrices, ROC curves, feature importance)

## Project Structure

```
ufc-prediction-2/
├── data/
│   ├── raw/              # Raw scraped fighter data (JSON files)
│   ├── processed/        # Cleaned data (CSV files)
│   └── features/         # Engineered features for ML
├── src/
│   ├── data_collection/
│   │   └── scraper.py    # Web scraping for UFC stats
│   ├── preprocessing/
│   │   └── cleaner.py    # Data cleaning and normalization
│   ├── features/
│   │   └── engineering.py # Feature engineering pipeline
│   ├── models/
│   │   ├── models.py     # Model definitions
│   │   └── trainer.py    # Model training pipeline
│   └── evaluation/
│       └── metrics.py    # Evaluation metrics and visualizations
├── config/
│   └── config.yaml       # Configuration file
├── models/               # Saved trained models
├── plots/                # Generated visualizations
├── requirements.txt      # Python dependencies
├── main.py              # Main pipeline orchestrator
├── scrape_all_fighters.py    # Batch scraping script
├── preprocess_data.py   # Preprocessing script
├── engineer_features.py # Feature engineering script
├── train_models.py      # Model training script
└── evaluate_models.py   # Model evaluation script
```

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd ufc-prediction-2
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories** (if they don't exist):
   ```bash
   mkdir -p data/raw data/processed data/features models plots logs
   ```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

This will execute all steps:
1. Preprocessing (if raw data exists)
2. Feature engineering
3. Model training
4. Model evaluation

### Update & Retrain (after new events)

To refresh fighter data and retrain models after new UFC events:

```bash
python update_and_retrain.py
```

This runs: **scrape** (with `--no-skip-existing`) → **preprocess** → **engineer** → **train**. Optional flags:

- `--num-events N` — Number of events for `--build-list` (default: 100)
- `--skip-scrape` — Skip scraping; only run preprocess → engineer → train (e.g. if you already re-scraped)

### Step-by-Step Execution

You can also run individual pipeline steps:

#### 1. Data Collection

Scrape fighter data from ufcstats.com:

```bash
# Build master list of fighters from events
python scrape_all_fighters.py --build-list --num-events 50

# Scrape all fighters from the master list
python scrape_all_fighters.py --scrape
```

**Note**: Scraping can take several hours depending on the number of fighters. Progress is automatically saved, so you can safely stop and resume the script at any time.

#### 2. Preprocessing

Clean and normalize scraped data:

```bash
python preprocess_data.py
```

This creates:
- `data/processed/all_fighters_stats.csv` - One row per fighter with aggregated stats
- `data/processed/all_fights.csv` - One row per fight

#### 3. Feature Engineering

Create ML-ready features:

```bash
python engineer_features.py
```

This creates:
- `data/features/fight_features.csv` - Feature matrix with 54 features per fight

#### 4. Model Training

Train all configured models:

```bash
python train_models.py
```

Models are saved to `models/` directory as `.pkl` files. Each run appends accuracy, F1, ROC-AUC (and timestamp) to `logs/training_metrics.json` for performance history tracking.

#### 5. Model Evaluation

Evaluate trained models:

```bash
python evaluate_models.py
```

This generates the following visualizations:
- Confusion matrices for each model
- ROC curves showing classification performance
- Feature importance plots (for tree-based models)
- Model comparison charts across multiple metrics

All plots are automatically saved to the `plots/` directory.

### Using the Main Pipeline

Run specific steps:
```bash
# Run only preprocessing and features
python main.py --steps preprocess features

# Skip certain steps
python main.py --skip-preprocessing --skip-features
```

## Configuration

Edit `config/config.yaml` to customize the following settings:

- **Data paths**: Where to save/load data
- **Scraping settings**: Rate limits, timeouts, retries
- **Model settings**: Which models to train, hyperparameters
- **Feature settings**: Which features to include
- **Evaluation settings**: Metrics to calculate, plots to generate

### Example: Enable/Disable Models

```yaml
models:
  random_forest:
    enabled: true
  xgboost:
    enabled: true
  svm:
    enabled: false  # Disable SVM (can be slow)
```

## Data Sources

- **Primary Source**: [ufcstats.com](http://ufcstats.com)
  - Fighter statistics
  - Fight history
  - Event listings

## Features

The system engineers 54 features per fight:

### Basic Features
- Win/Loss/Draw records
- Win percentages
- Total fights
- KO/TKO/Submission counts

### Physical Attributes
- Height, weight, reach
- Age differences
- Physical advantages

### Performance Metrics
- Striking accuracy and defense
- Strikes landed per minute
- Strikes absorbed per minute
- Takedown accuracy and defense
- Average takedowns per fight

### Advanced Features
- Recent form (last 3-5 fights)
- Win/loss streaks
- Average days between fights
- Strength of schedule (average opponent win percentage)
- Head-to-head comparisons (win pct advantage, striking advantage, striking differential, strength of schedule advantage, etc.)

## Models

The system supports multiple ML models:

- **Random Forest** - Ensemble tree-based model
- **XGBoost** - Gradient boosting (requires `xgboost` package)
- **LightGBM** - Light gradient boosting (requires `lightgbm` package)
- **Gradient Boosting** - Scikit-learn gradient boosting
- **Logistic Regression** - Linear classifier
- **SVM** - Support Vector Machine (can be slow)

All models include hyperparameter tuning via RandomizedSearchCV.

## Evaluation Metrics

Models are evaluated using:

- **Accuracy** - Overall prediction accuracy
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1 Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the ROC curve

Visualizations include:
- Confusion matrices
- ROC curves
- Feature importance plots
- Model comparison charts

## Project Status

✅ **Completed**:
- Data collection scraper
- Data preprocessing pipeline
- Feature engineering (54 features)
- Model training pipeline
- Model evaluation module
- Main pipeline orchestrator

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- UFC Stats data from ufcstats.com
- Scikit-learn for ML models
- Pandas for data processing

## Troubleshooting

### Import Errors

If you get import errors, make sure:
- Virtual environment is activated
- All dependencies are installed: `pip install -r requirements.txt`
- You're running scripts from the project root directory

### Scraping Issues

If scraping fails:
- Check internet connection
- Verify ufcstats.com is accessible
- Check rate limit settings in `config/config.yaml`
- Try scraping individual fighters first

### Model Training Fails

If model training fails:
- Verify feature matrix exists: `data/features/fight_features.csv`
- Check that XGBoost/LightGBM are installed if enabled
- Reduce `n_iter` in config if hyperparameter tuning is too slow
