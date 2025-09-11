import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


def parse_forecast_data(base_dir: str = ".") -> pd.DataFrame:
    """
    Read all JSON forecast files and combine into a single DataFrame.

    Args:
        base_dir: Root directory containing date-named folders

    Returns:
        DataFrame with consolidated forecast data
    """
    all_records = []

    # Find all date folders (assuming format YYYY-MM-DD)
    date_folders = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and len(d) == 10
        and d[4] == "-"
        and d[7] == "-"
    ]

    for folder in date_folders:
        folder_path = os.path.join(base_dir, folder)

        # Find all JSON files in the folder
        json_files = [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".json") and not f.startswith("TEST.")
        ]

        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extract model metadata
                organization = data.get("organization", "")
                model = data.get("model", "")

                # Process each forecast
                for forecast in data.get("forecasts", []):
                    record = {
                        "organization": organization,
                        "model": model,
                        "id": forecast.get("id"),
                        "forecast_due_date": forecast.get("forecast_due_date"),
                        "forecast": forecast.get("forecast"),
                        "resolved_to": forecast.get("resolved_to"),
                        "resolved": forecast.get("resolved"),
                        "market_value_on_due_date": forecast.get(
                            "market_value_on_due_date"
                        ),
                        "resolution_date": forecast.get("resolution_date"),
                        "source": forecast.get("source"),
                        "imputed": forecast.get("imputed"),
                        "direction": forecast.get("direction"),
                    }
                    all_records.append(record)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_records)
    return df


def parse_question_data(base_dir: str = ".") -> pd.DataFrame:
    """
    Parse JSON files and extract id, source, forecast_due_date,
    question_set, and freeze_datetime_value into a DataFrame.
    """
    data = []

    # Iterate through all JSON files in directory
    for json_file in Path(base_dir).glob("*.json"):
        with open(json_file, "r") as f:
            content = json.load(f)

        # Extract parent-level fields
        forecast_due_date = content.get("forecast_due_date")
        question_set = content.get("question_set")

        # Extract data from each question
        for question in content.get("questions", []):
            data.append(
                {
                    "id": question.get("id"),
                    "source": question.get("source"),
                    "forecast_due_date": forecast_due_date,
                    "question_set": question_set,
                    "freeze_datetime_value": question.get("freeze_datetime_value"),
                    "url": question.get("url"),
                }
            )

    return pd.DataFrame(data)


def process_parsed_data(
    df_forecasts: pd.DataFrame,
    df_questions: pd.DataFrame,
    df_model_release_dates: pd.DataFrame,
    imputation_threshold: float,
    reference_date: str = None,
) -> pd.DataFrame:
    """
    Process parsed forecast and question data.

    Args:
        df_forecasts: DataFrame with parsed forecast data
        df_questions: DataFrame with parsed question data
        imputation_threshold: Maximum allowed fraction of imputed forecasts per model

    Returns:
        Processed DataFrame
    """

    # Filtering
    print("Before filtering:{}".format(df_forecasts.shape))

    # Remove all combination questions (non-missing "direction")
    mask = df_forecasts["direction"].apply(lambda x: x == "[]")
    df_forecasts = df_forecasts.loc[mask,].copy()
    del df_forecasts["direction"]
    print("After removing combo questions:{}".format(df_forecasts.shape))

    # Remove all unresolved questions
    mask = df_forecasts["resolved"]
    df_forecasts = df_forecasts.loc[mask,].copy()
    print("After removing unresolved questions:{}".format(df_forecasts.shape))

    # Remove non-ForecastBench models with too many imputed questions
    df_temp = df_forecasts.groupby("model")["imputed"].mean().reset_index()
    if df_temp["model"].duplicated().sum() > 0:
        raise ValueError("There are duplicated models in the data.")
    df_temp.rename(columns={"imputed": "imputed_rate"}, inplace=True)
    df_forecasts = pd.merge(df_forecasts, df_temp, on="model", how="left")
    mask = (df_forecasts["imputed_rate"] <= imputation_threshold) | (
        df_forecasts["organization"] == "ForecastBench"
    )
    df_forecasts = df_forecasts.loc[mask,].copy()
    shape_after_filtering = df_forecasts.shape
    print(
        "After removing models with >5% imputed questions:{}".format(
            shape_after_filtering
        )
    )

    # Calculation of additional columns

    # Add a column indicating if the question is from a "market" source
    mask = df_forecasts["source"].apply(
        lambda x: x in ["infer", "manifold", "metaculus", "polymarket"]
    )
    df_forecasts["market_question"] = mask

    # Merge question data
    df_questions = df_questions[
        ["id", "forecast_due_date", "url", "freeze_datetime_value"]
    ].copy()
    df_questions.drop_duplicates(  # Deduplicate question data
        subset=["id", "forecast_due_date"], inplace=True
    )

    # Ensure consistent types for merging
    df_questions["forecast_due_date"] = pd.to_datetime(
        df_questions["forecast_due_date"]
    )
    df_forecasts["forecast_due_date"] = pd.to_datetime(
        df_forecasts["forecast_due_date"]
    )

    df_forecasts = pd.merge(
        df_forecasts, df_questions, on=["id", "forecast_due_date"], how="left"
    )
    df_forecasts = df_forecasts.rename(
        columns={"freeze_datetime_value": "market_value_on_freeze_date"}
    )

    # Create a unique question identifier by concatenating "id", "forecast_due_date",
    # and forecasting horizon
    df_forecasts["resolution_date"] = pd.to_datetime(df_forecasts["resolution_date"])
    df_forecasts["horizon"] = (
        df_forecasts["resolution_date"] - df_forecasts["forecast_due_date"]
    ).dt.days
    df_forecasts["question_id"] = (
        df_forecasts["id"].astype(str)
        + "_"
        + df_forecasts["forecast_due_date"].astype(str)
        + "_"
        + df_forecasts["horizon"].astype(str)
    )
    if df_forecasts[["question_id", "model"]].duplicated().sum() > 0:
        raise ValueError(
            "There are duplicated model forecasts for the same question and due date."
        )

    # Calculate Brier score
    df_forecasts["brier_score"] = (
        df_forecasts["forecast"] - df_forecasts["resolved_to"]
    ) ** 2

    # Add model release dates

    # Since models have various variants (e.g. "GPT-4o (zero shot)"
    # and "GPT-4o (zero shot with freeze values)"), we first
    # extract the base model name
    df_forecasts["model_suffix"] = (
        df_forecasts["model"]
        .str.extract(r"^([^(]+)\s*\(")[0]
        .fillna(df_forecasts["model"])
        .str.strip()
    )
    df_model_release_dates.rename(
        columns={"model": "model_suffix", "release_date": "model_release_date"},
        inplace=True,
    )
    df_forecasts = pd.merge(
        df_forecasts,
        df_model_release_dates,
        on="model_suffix",
        how="left",
    )

    # Calculate days since model release
    df_forecasts["model_days_released"] = (
        pd.to_datetime(df_forecasts["forecast_due_date"])
        - pd.to_datetime(df_forecasts["model_release_date"])
    ).dt.days

    # Calculate day of first model forecast
    # in ForecastBench (i.e., start of participation)
    # separately for dataset & market questions
    if reference_date is None:
        reference_date = pd.to_datetime(df_forecasts["resolution_date"]).max()

    for name, mask_val in [
        ("market", True),
        ("dataset", False),
    ]:
        # Calculate date of first forecast
        mask = df_forecasts["market_question"] == mask_val
        df_temp = (
            df_forecasts[mask].groupby("model")["forecast_due_date"].min().reset_index()
        )
        df_temp.rename(
            columns={"forecast_due_date": f"model_first_forecast_date_{name}"},
            inplace=True,
        )
        df_forecasts = pd.merge(df_forecasts, df_temp, on="model", how="left")

        # Calculate number of days since first forecast
        # relative to the reference date
        df_forecasts[f"model_days_active_{name}"] = (
            pd.to_datetime(reference_date)
            - pd.to_datetime(df_forecasts[f"model_first_forecast_date_{name}"])
        ).dt.days
    df_forecasts["model_first_forecast_date"] = np.minimum(
        df_forecasts["model_first_forecast_date_market"],
        df_forecasts["model_first_forecast_date_dataset"],
    )

    # Check if all models have release dates
    mask = df_forecasts["organization"] != "ForecastBench"
    if df_forecasts.loc[mask, "model_release_date"].isnull().any():
        missing_models = (
            df_forecasts.loc[
                mask & df_forecasts["model_release_date"].isnull(), "model"
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"The following models are missing release dates: {missing_models}"
        )

    # Select key columns & export
    df_forecasts = df_forecasts[
        [
            "organization",
            "model",
            "model_release_date",
            "model_first_forecast_date",
            "model_first_forecast_date_market",
            "model_first_forecast_date_dataset",
            "model_days_released",
            "model_days_active_market",
            "model_days_active_dataset",
            "imputed_rate",
            "imputed",
            "question_id",
            "market_question",
            "source",
            "url",
            "forecast_due_date",
            "resolution_date",
            "horizon",
            "forecast",
            "resolved_to",
            "brier_score",
            "market_value_on_freeze_date",
            "market_value_on_due_date",
        ]
    ].copy()
    shape_after_extra_fields = df_forecasts.shape
    if shape_after_filtering[0] != shape_after_extra_fields[0]:
        raise ValueError("The number of rows changed after adding extra fields.")

    return df_forecasts


def get_diff_adj_brier(
    df: pd.DataFrame, max_model_days_released: int, drop_baseline_models: list
):
    df = df.copy()
    df_fe_model = df.copy()

    # Data filtering for 2FE estimation

    # Remove old models
    mask = (df_fe_model["model_days_released"] < max_model_days_released) | (
        df_fe_model["organization"] == "ForecastBench"
    )
    df_fe_model = df_fe_model[mask].copy()

    # Remove benchmark models
    mask = ~df_fe_model["model"].isin(drop_baseline_models)
    df_fe_model = df_fe_model[mask].copy()

    # Estimate the 2FE model
    mod = pf.feols("brier_score ~ 1 | question_id + model", data=df_fe_model)
    dict_question_fe = mod.fixef()["C(question_id)"]

    if len(dict_question_fe) != len(df["question_id"].unique()):
        raise ValueError(
            f"Estimated num. of question fixed effects \
                  ({len(dict_question_fe)}) not equal to num. "
            f"of questions ({len(df['question_id'].unique())})"
        )

    df["question_fe"] = df["question_id"].map(dict_question_fe)
    df["diff_adj_brier_score"] = df["brier_score"] - df["question_fe"]
    return df


def compute_diff_adj_scores(
    df: pd.DataFrame, max_model_days_released: int, drop_baseline_models: list
):
    """Compute diff-adj Brier scores for all questions"""
    df = df.copy()

    # Process market and dataset questions separately
    mask = df["market_question"]

    # Market questions
    df_market = get_diff_adj_brier(
        df=df[mask],
        max_model_days_released=max_model_days_released,
        drop_baseline_models=drop_baseline_models,
    )

    # Dataset questions
    df_dataset = get_diff_adj_brier(
        df=df[~mask],
        max_model_days_released=max_model_days_released,
        drop_baseline_models=drop_baseline_models,
    )

    # Combine back
    df_combined = pd.concat([df_market, df_dataset], ignore_index=True)

    return df_combined


def create_leaderboard(
    df_with_scores: pd.DataFrame,
    min_days_active_market: int = None,
    min_days_active_dataset: int = None,
):
    """Aggregate diff-adj scores into leaderboard with
     (potentially) activity filters.

    Args:
        df_with_scores: DataFrame with diff-adj scores
        min_days_active_market: Min days active to show market scores
        min_days_active_dataset: Min days active to show dataset scores"""
    df = df_with_scores.copy()

    # Create base dataframe
    df_res = (
        df[
            [
                "model",
                "model_days_active_market",
                "model_days_active_dataset",
                "organization",
            ]
        ]
        .copy()
        .drop_duplicates()
    )

    # Aggregate by question type
    for name, mask_val, min_days in [
        ("market", True, min_days_active_market),
        ("dataset", False, min_days_active_dataset),
    ]:
        df_subset = df[df["market_question"] == mask_val]

        df_agg = (
            df_subset.groupby("model")["diff_adj_brier_score"]
            .agg([("diff_adj_brier_score", "mean"), ("n_forecasts", "count")])
            .reset_index()
        )

        # Apply activity filter only if min_days is specified
        if min_days is not None:
            df_agg = pd.merge(
                df_agg,
                df_res[["model", f"model_days_active_{name}", "organization"]],
                on="model",
                how="left",
            )

            # Activity filter not specified for ForecastBench models
            # since they do not have a release date
            mask_active = (df_agg[f"model_days_active_{name}"] >= min_days) | (
                df_agg["organization"] == "ForecastBench"
            )
            df_agg.loc[~mask_active, "diff_adj_brier_score"] = np.nan

        df_agg.rename(
            columns={
                "diff_adj_brier_score": f"diff_adj_brier_score_{name}",
                "n_forecasts": f"n_forecasts_{name}",
            },
            inplace=True,
        )

        df_res = pd.merge(
            df_res,
            df_agg[["model", f"diff_adj_brier_score_{name}", f"n_forecasts_{name}"]],
            on="model",
            how="left",
        )

    # Normalize to "Always 0.5" baseline
    for name in ["market", "dataset"]:
        mask = df_res["model"] == "Always 0.5"
        reference_value = df_res.loc[mask, f"diff_adj_brier_score_{name}"].values[0]

        # Only normalize non-null values
        not_null = df_res[f"diff_adj_brier_score_{name}"].notna()
        df_res.loc[not_null, f"diff_adj_brier_score_{name}"] = 0.25 + (
            df_res.loc[not_null, f"diff_adj_brier_score_{name}"] - reference_value
        )

    # Weighted average - only if both components are available
    mask_both = (
        df_res["diff_adj_brier_score_market"].notna()
        & df_res["diff_adj_brier_score_dataset"].notna()
    )

    df_res.loc[mask_both, "diff_adj_brier_score"] = (
        0.5 * df_res.loc[mask_both, "diff_adj_brier_score_market"]
        + 0.5 * df_res.loc[mask_both, "diff_adj_brier_score_dataset"]
    )

    df_res = df_res.sort_values(by="diff_adj_brier_score", ascending=True)

    return df_res
