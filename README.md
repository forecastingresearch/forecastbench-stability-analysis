# ForecastBench Stability Analysis

This repository contains tools for analyzing the stability of leaderboards on ForecastBench. The analysis calculates difficulty-adjusted Brier scores and performs various stability metrics to assess how consistent model rankings are over time.

## Project Structure

```
forecastbench-stability-analysis/
├── src/
│   └── stability_analysis.py    # Core analysis functions and classes
├── data/
│   ├── raw/
│   │   ├── forecast_sets/       # Forecast JSON files organized by date
│   │   ├── question_sets/       # Question metadata JSON files  
│   │   └── model_release_dates.csv
│   ├── processed/               # Intermediate processed data
│   └── results/                 # Analysis outputs and visualizations
├── run_analysis.py             # Main analysis script
├── generate_html_viewer.py     # Interactive results viewer generator
└── notebooks/                  # Development notebooks
```

## Installation

### Prerequisites

- Python 3.8+
- Required packages:

```bash
pip install pandas numpy matplotlib pyfixest
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd forecastbench-stability-analysis
```

2. Ensure your data structure matches the expected format:
   - Forecast JSON files in `data/raw/forecast_sets/YYYY-MM-DD/`
   - Question JSON files in `data/raw/question_sets/`
   - Model release dates in `data/raw/model_release_dates.csv`

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python run_analysis.py
```

This will:
1. Parse forecast and question JSON files
2. Process and merge the data
3. Calculate difficulty-adjusted Brier scores  
4. Generate multiple leaderboard configurations
5. Perform stability and sample size analyses
6. Create an interactive HTML leaderboard viewer

**Expected runtime**: 1-2 minutes on a standard laptop

### Configuration

Key parameters can be modified in `run_analysis.py`:

```python
# Analysis parameters
REFERENCE_DATE = "2025-09-10"           # Reference date for calculations
IMPUTATION_THRESHOLD = 0.05             # Max fraction of imputed forecasts
MAX_MODEL_DAYS_RELEASED = 365           # Model age limit for estimation
STABILITY_THRESHOLDS = [100, 180]       # Days active for stability analysis

# Stability metrics to calculate
STABILITY_METRICS = [
    "correlation",                      # Pearson correlation
    "rank_correlation",                 # Spearman rank correlation  
    "median_displacement",              # Median rank displacement
    "top25_retention",                  # Top-25% retention rate
]
```

### Programmatic Usage

You can also use the analysis functions directly:

```python
from src.stability_analysis import (
    parse_forecast_data, 
    process_parsed_data,
    compute_diff_adj_scores,
    create_leaderboard,
    perform_stability_analysis
)

# Parse forecast data
df_forecasts = parse_forecast_data("./data/raw/forecast_sets/")

# Calculate difficulty-adjusted scores
df_with_scores = compute_diff_adj_scores(
    df=df_forecasts,
    max_model_days_released=365,
    drop_baseline_models=["Always 0.5", "Random Uniform"]
)

# Create leaderboard
df_leaderboard = create_leaderboard(
    df_with_scores,
    min_days_active_market=50,
    min_days_active_dataset=50
)

# Perform stability analysis
perform_stability_analysis(
    df_with_scores=df_with_scores,
    model_days_active_threshold=100,
    results_folder="./results",
    metrics=["correlation", "rank_correlation"]
)
```

## Output Files

### Results Directory Structure

After running the analysis, you'll find:

```
data/results/
├── leaderboard_*.csv                    # Various leaderboard configurations
├── stability_*_threshold_*.csv          # Stability analysis results
├── sample_size_analysis_*.csv           # Sample size analysis results  
├── stability_*_*.png                    # Stability metric visualizations
├── sample_size_analysis_*.png           # Sample size plots
└── leaderboard_viewer.html              # Interactive results viewer
```

### Key Output Files

- **Leaderboards**: `leaderboard_baseline.csv`, `leaderboard_proposal.csv`, etc.
- **Stability Metrics**: Stability metrics at various thresholds
- **Sample Size Analysis**: Days needed for models to reach question thresholds
- **Interactive Viewer**: HTML interface for exploring all results

## Data Format Requirements

### Forecast JSON Structure
```json
{
  "organization": "ModelProvider",
  "model": "Model Name",
  "forecasts": [
    {
      "id": "question_id",
      "forecast_due_date": "YYYY-MM-DD",
      "forecast": 0.75,
      "resolved_to": 1,
      "resolved": true,
      "resolution_date": "YYYY-MM-DD",
      "source": "metaculus",
      "imputed": false
    }
  ]
}
```

### Question JSON Structure
```json
{
  "forecast_due_date": "YYYY-MM-DD",
  "question_set": "llm",
  "questions": [
    {
      "id": "question_id", 
      "source": "metaculus",
      "freeze_datetime_value": 0.65,
      "url": "https://..."
    }
  ]
}
```

### Model Release Dates CSV
```csv
model,release_date
GPT-4o,2024-05-13
Claude-3.5-Sonnet,2024-06-20
```

## Methodology 

### Stability Assessment

For each stability threshold (e.g., 100 days), the analysis:

1. Calculates baseline scores using complete forecasting records
2. For each day X from 0 to threshold:
   - Filters to forecasts made within first X days of model activity
   - Calculates incomplete-period scores
   - Computes stability metrics comparing incomplete vs. complete rankings

### Sample Size Analysis

Determines the number of days needed for 80% of models to reach various resolved question thresholds, helping inform data requirements for reliable model evaluation.