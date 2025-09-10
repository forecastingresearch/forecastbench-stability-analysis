#!/usr/bin/env python
import sys

sys.path.append("src")

import pandas as pd

from stability_analysis import (
    compute_diff_adj_scores,
    create_leaderboard,
    parse_forecast_data,
    parse_question_data,
    process_parsed_data,
)

# EXPECTED RUNTIME: 1-2 minutes on a standard laptop

# =====================================================
# GLOBAL CONFIGURATION
# =====================================================
IMPUTATION_THRESHOLD = 0.05  # Maximum allowed fraction of imputed forecasts per model
MAX_DAYS_SINCE_RELEASE = (
    365  # Maximum number of days since model release to be included
)
# in the 2FE model estimation
DROP_BASELINE_MODELS = [
    "Always 0.5",  # Models excluded from 2FE estimation
    "Always 0",
    "Always 1",
    "Random Uniform",
    "Imputed Forecaster",
]

RAW_FOLDER = "./data/raw"
PROCESSED_FOLDER = "./data/processed"
RESULTS_FOLDER = "./data/results"

# =====================================================
# MAIN SCRIPT
# =====================================================


def main():
    # print("Parsing forecast JSON files...", end="", flush=True)
    # df = parse_forecast_data(f"{RAW_FOLDER}/forecast_sets/")
    # df.to_csv(f"{PROCESSED_FOLDER}/parsed_forecasts.csv", index=False)
    # print(" ✅")
    # print("Parsing question JSON files...", end="", flush=True)
    # df = parse_question_data(f"{RAW_FOLDER}/question_sets/")
    # df.to_csv(f"{PROCESSED_FOLDER}/parsed_questions.csv", index=False)
    # print(" ✅")
    # print("Processing parsed files...", end="", flush=True)
    # df_forecasts = pd.read_csv(f"{PROCESSED_FOLDER}/parsed_forecasts.csv")
    # df_questions = pd.read_csv(f"{PROCESSED_FOLDER}/parsed_questions.csv")
    # df_model_release_dates = pd.read_csv(f"{RAW_FOLDER}/model_release_dates.csv")
    # df = process_parsed_data(
    #     df_forecasts=df_forecasts,
    #     df_questions=df_questions,
    #     df_model_release_dates=df_model_release_dates,
    #     imputation_threshold=IMPUTATION_THRESHOLD,
    # )
    # df.to_csv(f"{PROCESSED_FOLDER}/processed_data.csv", index=False)
    # df[["model", "model_release_date"]].drop_duplicates().sort_values(
    #     by="model", ascending=True
    # ).to_csv(f"{PROCESSED_FOLDER}/model_release_dates.csv", index=False)
    # print(" ✅")
    print("Estimating diff-adj Brier scores...", end="", flush=True)

    # Baseline leaderboard
    df = pd.read_csv(f"{PROCESSED_FOLDER}/processed_data.csv")
    mask = (pd.to_datetime(df["forecast_due_date"]) < "2025-06-01") & (
        (pd.to_datetime(df["model_first_forecast_date"]) < "2025-06-01")
        | (df["organization"] == "ForecastBench")
    )
    df = df.loc[mask].copy()

    df_with_scores = compute_diff_adj_scores(
        df=df,
        max_days_since_release=MAX_DAYS_SINCE_RELEASE,
        drop_baseline_models=DROP_BASELINE_MODELS,
    )
    df_leaderboard = create_leaderboard(df_with_scores)
    df_leaderboard.to_csv(f"{RESULTS_FOLDER}/leaderboard_baseline.csv", index=False)

    # Use all resolved questions
    df = pd.read_csv(f"{PROCESSED_FOLDER}/processed_data.csv")
    mask = (pd.to_datetime(df["model_first_forecast_date"]) < "2025-06-01") | (
        df["organization"] == "ForecastBench"
    )
    df = df.loc[mask].copy()

    df_with_scores = compute_diff_adj_scores(
        df=df,
        max_days_since_release=MAX_DAYS_SINCE_RELEASE,
        drop_baseline_models=DROP_BASELINE_MODELS,
    )
    df_leaderboard = create_leaderboard(df_with_scores)
    df_leaderboard.to_csv(f"{RESULTS_FOLDER}/leaderboard_all_resolved.csv", index=False)

    # Use all resolved questions and models
    df = pd.read_csv(f"{PROCESSED_FOLDER}/processed_data.csv")

    df_with_scores = compute_diff_adj_scores(
        df=df,
        max_days_since_release=MAX_DAYS_SINCE_RELEASE,
        drop_baseline_models=DROP_BASELINE_MODELS,
    )
    df_leaderboard = create_leaderboard(df_with_scores)
    df_leaderboard.to_csv(f"{RESULTS_FOLDER}/leaderboard_full.csv", index=False)

    print(" ✅")


if __name__ == "__main__":
    main()
