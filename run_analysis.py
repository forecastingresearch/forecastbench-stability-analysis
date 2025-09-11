#!/usr/bin/env python
import sys

sys.path.append("src")

import pandas as pd

from generate_html_viewer import generate_html_viewer
from stability_analysis import (
    compute_diff_adj_scores,
    create_leaderboard,
    parse_forecast_data,
    parse_question_data,
    perform_sample_size_analysis,
    perform_stability_analysis,
    process_parsed_data,
)


def generate_leaderboard(
    df,
    mask,
    output_filename,
    max_model_days_released,
    drop_baseline_models,
    results_folder,
    min_days_active_market=None,
    min_days_active_dataset=None,
    return_scored_data=False,
):
    """Generate and save a leaderboard with given filters"""
    df_filtered = df.loc[mask].copy() if mask is not None else df.copy()

    df_with_scores = compute_diff_adj_scores(
        df=df_filtered,
        max_model_days_released=max_model_days_released,
        drop_baseline_models=drop_baseline_models,
    )
    df_leaderboard = create_leaderboard(
        df_with_scores,
        min_days_active_market=min_days_active_market,
        min_days_active_dataset=min_days_active_dataset,
    )
    df_leaderboard.to_csv(f"{results_folder}/{output_filename}", index=False)
    if return_scored_data:
        return {"df_leaderboard": df_leaderboard, "df_with_scores": df_with_scores}
    else:
        return {"df_leaderboard": df_leaderboard}


# EXPECTED RUNTIME: 1-2 minutes on a standard laptop

# =====================================================
# GLOBAL CONFIGURATION
# =====================================================

# Date of "now" for calculating days active on ForecastBench
REFERENCE_DATE = "2025-09-10"

# Maximum allowed fraction of imputed forecasts per model
IMPUTATION_THRESHOLD = 0.05

# Max number of days since model release to be included
# in the 2FE model estimation
MAX_MODEL_DAYS_RELEASED = 365

# Models excluded from 2FE estimation
DROP_BASELINE_MODELS = [
    "Always 0.5",
    "Always 0",
    "Always 1",
    "Random Uniform",
    "Imputed Forecaster",
]

# Number of days active for models for the stability analysis
STABILITY_THRESHOLDS = [100, 180]

# Stability metrics to calculate
STABILITY_METRICS = [
    "correlation",  # Pearson correlation
    "rank_correlation",  # Spearman rank correlation
    "median_displacement",  # Median absolute rank displacement
    "top25_retention",  # Top-25% retention rate
]

# Stability analysis visualization options
STABILITY_VIZ_CONFIG = {
    "save_individual_plots": True,  # Save individual metric plots
    "save_combined_plot": True,  # Save combined subplot
    "show_plots": False,  # Display plots during execution
    "dpi": 300,  # Plot resolution
    "figsize": (12, 10),  # Combined plot size
}

# Metric display labels for visualization
METRIC_LABELS = {
    "correlation": "Score Correlation",
    "rank_correlation": "Rank Correlation (Spearman ρ)",
    "median_displacement": "Median Rank Displacement",
    "top25_retention": "Top-25% Retention Rate",
}

# Folder definitions
RAW_FOLDER = "./data/raw"
PROCESSED_FOLDER = "./data/processed"
RESULTS_FOLDER = "./data/results"
GRAPH_FOLDER = "./data/results/graphs"

# =====================================================
# MAIN SCRIPT
# =====================================================


def main():
    print("Parsing forecast JSON files...", end="", flush=True)
    df = parse_forecast_data(f"{RAW_FOLDER}/forecast_sets/")
    df.to_csv(f"{PROCESSED_FOLDER}/parsed_forecasts.csv", index=False)
    print(" ✅")
    print("Parsing question JSON files...", end="", flush=True)
    df = parse_question_data(f"{RAW_FOLDER}/question_sets/")
    df.to_csv(f"{PROCESSED_FOLDER}/parsed_questions.csv", index=False)
    print(" ✅")
    print("Processing parsed files...", end="", flush=True)
    df_forecasts = pd.read_csv(f"{PROCESSED_FOLDER}/parsed_forecasts.csv")
    df_questions = pd.read_csv(f"{PROCESSED_FOLDER}/parsed_questions.csv")
    df_model_release_dates = pd.read_csv(f"{RAW_FOLDER}/model_release_dates.csv")
    df = process_parsed_data(
        df_forecasts=df_forecasts,
        df_questions=df_questions,
        df_model_release_dates=df_model_release_dates,
        imputation_threshold=IMPUTATION_THRESHOLD,
        reference_date=REFERENCE_DATE,
    )
    df.to_csv(f"{PROCESSED_FOLDER}/processed_data.csv", index=False)
    df[["model", "model_release_date"]].drop_duplicates().sort_values(
        by="model", ascending=True
    ).to_csv(f"{PROCESSED_FOLDER}/model_release_dates.csv", index=False)
    print(" ✅")
    print("Estimating diff-adj Brier scores...", end="", flush=True)
    df = pd.read_csv(f"{PROCESSED_FOLDER}/processed_data.csv")

    # Define leaderboard configurations
    leaderboard_config = [
        {
            "name": "leaderboard_baseline.csv",
            "mask": (df["forecast_due_date"] < "2025-06-01")
            & (
                (df["model_first_forecast_date"] < "2025-06-01")
                | (df["organization"] == "ForecastBench")
            ),
            "min_days_active_market": None,
            "min_days_active_dataset": None,
            "stability_analysis": True,
            "stability_metrics": STABILITY_METRICS,
            "sample_size_analysis": True,
        },
        {
            "name": "leaderboard_baseline_all_data_for_2FE.csv",
            "mask": None,
            "min_days_active_market": 100,
            "min_days_active_dataset": 100,
            "stability_analysis": False,
            "sample_size_analysis": False,
        },
        {
            "name": "leaderboard_all_data_no_filtering.csv",
            "mask": None,
            "min_days_active_market": None,
            "min_days_active_dataset": None,
            "stability_analysis": False,
            "sample_size_analysis": False,
        },
        {
            "name": "leaderboard_proposal.csv",
            "mask": None,
            "min_days_active_market": 50,
            "min_days_active_dataset": 7,
            "stability_analysis": False,
            "sample_size_analysis": False,
        },
    ]

    # Generate all leaderboards
    for config in leaderboard_config:
        res_dict = generate_leaderboard(
            df=df,
            mask=config["mask"],
            output_filename=config["name"],
            max_model_days_released=MAX_MODEL_DAYS_RELEASED,
            drop_baseline_models=DROP_BASELINE_MODELS,
            results_folder=RESULTS_FOLDER,
            min_days_active_market=config["min_days_active_market"],
            min_days_active_dataset=config["min_days_active_dataset"],
            return_scored_data=config["stability_analysis"]
            or config.get("sample_size_analysis", False),
        )
        if config["stability_analysis"]:
            # Run stability analysis for each threshold value
            for threshold in STABILITY_THRESHOLDS:
                perform_stability_analysis(
                    df_with_scores=res_dict["df_with_scores"],
                    model_days_active_threshold=threshold,
                    results_folder=RESULTS_FOLDER,
                    metrics=STABILITY_METRICS,
                    viz_config=STABILITY_VIZ_CONFIG,
                    metric_labels=METRIC_LABELS,
                    output_suffix=f"{config['name'].replace('.csv', '')}_threshold_{threshold}",
                )

        if config.get("sample_size_analysis", False):
            # Run sample size analysis
            perform_sample_size_analysis(
                df_with_scores=res_dict["df_with_scores"],
                results_folder=RESULTS_FOLDER,
                output_suffix=config["name"].replace(".csv", ""),
            )

    print(" ✅")

    print("Generating interactive HTML viewer...", end="", flush=True)
    html_viewer_path = generate_html_viewer(f"{RESULTS_FOLDER}/leaderboard_viewer.html")
    print(" ✅")
    print(f"HTML viewer saved to: {html_viewer_path}")


if __name__ == "__main__":
    main()
