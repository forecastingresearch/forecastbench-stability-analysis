import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf


class StabilityMetrics:
    """Calculator for various stability metrics"""

    def __init__(
        self, df_full_scores: pd.DataFrame, df_incomplete_scores: pd.DataFrame
    ):
        """
        Initialize with full and incomplete score dataframes

        Args:
            df_full_scores: DataFrame with full period scores
                (has 'diff_adj_brier_score' column)
            df_incomplete_scores: DataFrame with incomplete period scores
                (has 'diff_adj_brier_score' column)
        """
        self.df_full = df_full_scores.copy()
        self.df_incomplete = df_incomplete_scores.copy()
        self.df_combined = self._merge_scores()

    def _merge_scores(self) -> pd.DataFrame:
        """Merge full and incomplete score dataframes"""
        df_incomplete_renamed = self.df_incomplete.rename(
            columns={"diff_adj_brier_score": "diff_adj_brier_score_0_x"}
        )
        return pd.merge(
            self.df_full, df_incomplete_renamed, on="model", how="left", validate="1:1"
        )

    def score_correlation(self) -> float:
        """Pearson correlation between full and incomplete period scores"""
        return (
            self.df_combined[["diff_adj_brier_score", "diff_adj_brier_score_0_x"]]
            .corr()
            .values[0, 1]
        )

    def rank_correlation(self) -> float:
        """Spearman rank correlation between rankings"""
        return (
            self.df_combined[["diff_adj_brier_score", "diff_adj_brier_score_0_x"]]
            .corr(method="spearman")
            .values[0, 1]
        )

    def median_displacement(self) -> float:
        """Median absolute rank displacement"""
        full_ranks = self.df_combined["diff_adj_brier_score"].rank(ascending=True)
        incomplete_ranks = self.df_combined["diff_adj_brier_score_0_x"].rank(
            ascending=True
        )
        return np.median(np.abs(full_ranks - incomplete_ranks))

    def top_k_retention(self, k: int = 25) -> float:
        """
        Top-K retention rate (fraction of top-K models that remain in top-K)

        Args:
            k: Number or percentage of top models to consider
        """
        n_models = len(self.df_combined)

        # If k > n_models, use 25% of models
        if k > n_models:
            raise ValueError(
                f"Number of top-k model ({k}) exceeds the number "
                f"of models in the sample ({n_models})"
            )

        # Get top-k models by score (nsmallest because lower scores are better)
        full_top_k = set(self.df_combined.nsmallest(k, "diff_adj_brier_score")["model"])
        incomplete_top_k = set(
            self.df_combined.nsmallest(k, "diff_adj_brier_score_0_x")["model"]
        )

        # Calculate retention rate
        if len(full_top_k) == 0:
            return 0.0
        return len(full_top_k.intersection(incomplete_top_k)) / len(full_top_k)


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
                model_organization = data.get("model_organization", "")
                model = data.get("model", "")

                # Process each forecast
                for forecast in data.get("forecasts", []):
                    record = {
                        "model_organization": model_organization,
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

    Args:
        base_dir: Directory containing JSON question files (default: current directory)

    Returns:
        DataFrame with parsed question data including id, source, forecast_due_date,
        question_set, freeze_datetime_value, and url columns
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
    reference_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process parsed forecast and question data.

    Args:
        df_forecasts: DataFrame with parsed forecast data
        df_questions: DataFrame with parsed question data
        df_model_release_dates: DataFrame with model release dates
        imputation_threshold: Maximum allowed fraction of imputed forecasts per model
        reference_date: Reference date for calculating model activity (default: max
            resolution date)

    Returns:
        Processed DataFrame with additional calculated columns including Brier scores,
        model activity metrics, and consolidated forecast information
    """

    # Filtering
    print("Before filtering:{}".format(df_forecasts.shape))

    # Add a column indicating if the question is from a "market" source
    mask = df_forecasts["source"].isin(["infer", "manifold", "metaculus", "polymarket"])
    df_forecasts["market_question"] = mask

    # Remove non-ForecastBench models with >imputation_threshold imputed
    # questions. Filtering is done at the forecast_due_date level (i.e.,
    # round-level), and separately for dataset & market questions

    for name, mask_val in [
        ("market", True),
        ("dataset", False),
    ]:
        # Calculate imputation rate per model per round
        mask = df_forecasts["market_question"] == mask_val
        df_imputation_rate = (
            df_forecasts[mask]
            .groupby(["forecast_due_date", "model"])["imputed"]
            .mean()
            .reset_index()
            .rename(columns={"imputed": f"round_imputed_rate_{name}"})
        )

        # Merge back and filter
        df_forecasts = pd.merge(
            df_forecasts,
            df_imputation_rate,
            how="left",
            on=["forecast_due_date", "model"],
        )

    # Apply filtering. Remove a model if the imputation
    # threshold is exceeded for either market or dataset questions
    mask = (df_forecasts["round_imputed_rate_market"] <= imputation_threshold) & (
        df_forecasts["round_imputed_rate_dataset"] <= imputation_threshold
    )
    df_forecasts = df_forecasts[mask]

    print(
        f"After removing models with >{imputation_threshold*100}% "
        f"imputed: {df_forecasts.shape}"
    )

    # Remove all combination questions (non-missing "direction")
    mask = df_forecasts["direction"] == "[]"
    df_forecasts = df_forecasts.loc[mask,].copy()
    del df_forecasts["direction"]
    print("After removing combo questions:{}".format(df_forecasts.shape))

    # Remove all unresolved questions
    mask = df_forecasts["resolved"]
    df_forecasts = df_forecasts.loc[mask,].copy()
    print("After removing unresolved questions:{}".format(df_forecasts.shape))

    shape_after_filtering = df_forecasts.shape

    # Calculation of additional columns

    # Merge question data
    df_questions = df_questions[
        ["id", "source", "forecast_due_date", "url", "freeze_datetime_value"]
    ].copy()

    # Deduplicate question data. Selecting
    # "source" to avoid the edge case that two
    # different sources use the same id
    df_questions.drop_duplicates(
        subset=["id", "source", "forecast_due_date"], inplace=True
    )

    # Ensure consistent types for merging
    df_questions["forecast_due_date"] = pd.to_datetime(
        df_questions["forecast_due_date"]
    )
    df_forecasts["forecast_due_date"] = pd.to_datetime(
        df_forecasts["forecast_due_date"]
    )

    df_forecasts = pd.merge(
        df_forecasts, df_questions, on=["id", "source", "forecast_due_date"], how="left"
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
        df_forecasts["source"].astype(str)
        + "_"
        + df_forecasts["id"].astype(str)
        + "_"
        + df_forecasts["forecast_due_date"].astype(str)
        + "_"
        + df_forecasts["horizon"].astype(str)
    )
    if df_forecasts[["question_id", "model"]].duplicated().sum() > 0:
        raise ValueError(
            "There are duplicated model forecasts for the same source, "
            "question, and due date."
        )

    # Calculate Brier score
    df_forecasts["brier_score"] = (
        df_forecasts["forecast"] - df_forecasts["resolved_to"]
    ) ** 2

    # Calculate market Brier score
    df_forecasts["market_brier_score"] = (
        df_forecasts["market_value_on_due_date"] - df_forecasts["resolved_to"]
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
    df_forecasts["model_days_active"] = np.maximum(
        df_forecasts["model_days_active_market"],
        df_forecasts["model_days_active_dataset"],
    )

    # Check if all models have release dates
    mask = df_forecasts["model_organization"] != "ForecastBench"
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
            "model_organization",
            "model",
            "model_release_date",
            "model_first_forecast_date",
            "model_first_forecast_date_market",
            "model_first_forecast_date_dataset",
            "model_days_released",
            "model_days_active",
            "model_days_active_market",
            "model_days_active_dataset",
            "round_imputed_rate_dataset",
            "round_imputed_rate_market",
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
            "market_brier_score",
        ]
    ].copy()
    shape_after_extra_fields = df_forecasts.shape
    if shape_after_filtering[0] != shape_after_extra_fields[0]:
        raise ValueError("The number of rows changed after adding extra fields.")

    return df_forecasts


def get_diff_adj_brier(
    df: pd.DataFrame,
    max_model_days_released: int,
    drop_baseline_models: list,
    mkt_adj_weight: float = 1.0,
    exclude_tournament_models_in_2FE: bool = False,
) -> pd.DataFrame:
    """
    Calculate difficulty-adjusted Brier scores using two-way fixed effects estimation.

    Args:
        df: DataFrame with forecast data including brier_score, question_id, and model
            columns
        max_model_days_released: Maximum days since model release for inclusion
        drop_baseline_models: List of baseline model names to exclude from estimation
        mkt_adj_weight: If only market questions are present, weight given for market
            Brier score when calculating question difficulties. Question difficulties
            (i.e., question fixed effects) are calculated as
            mkt_adj_weight * market_brier + (1 - mkt_adj_weight) * 2FE_estimate
        exclude_tournament_models_in_2FE: If true, exclude tournament models from the
            estimation of question FE

    Returns:
        DataFrame with added question_fe and diff_adj_brier_score columns

    Raises:
        ValueError: If estimated question fixed effects don't match expected number of
            questions
    """
    df = df.copy()
    df_fe_model = df.copy()

    if mkt_adj_weight < 0.0 or mkt_adj_weight > 1.0:
        raise ValueError(
            f"Market weight should be in [0, 1] but instead equals {mkt_adj_weight}"
        )

    # Data filtering for 2FE estimation

    # Remove old models
    mask = (df_fe_model["model_days_released"] < max_model_days_released) | (
        df_fe_model["model_organization"] == "ForecastBench"
    )
    df_fe_model = df_fe_model[mask].copy()

    # Remove benchmark models
    mask = ~df_fe_model["model"].isin(drop_baseline_models)
    df_fe_model = df_fe_model[mask].copy()

    # Remove tournament models, if selected
    if exclude_tournament_models_in_2FE:
        mask = df_fe_model["model"].apply(lambda x: "freeze" not in x) & df_fe_model[
            "model"
        ].apply(lambda x: "news" not in x)
        df_fe_model = df_fe_model[mask].copy()

    # Estimate the 2FE model
    mod = pf.feols("brier_score ~ 1 | question_id + model", data=df_fe_model)
    dict_question_fe = mod.fixef()["C(question_id)"]

    if len(dict_question_fe) != len(df["question_id"].unique()):
        raise ValueError(
            f"Estimated num. of question fixed effects"
            f" ({len(dict_question_fe)}) not equal to num."
            f" of questions ({len(df['question_id'].unique())})"
        )

    df["question_fe"] = df["question_id"].map(dict_question_fe)

    # For market-questions only, perform the market Brier
    # adjustment
    if mkt_adj_weight > 0.0:
        question_types = df["market_question"].unique()
        is_market_only = (len(question_types) == 1) and question_types[0]
        if is_market_only:
            # Check for missing values
            if df["market_brier_score"].isna().any():
                raise ValueError("Some questions missing market Brier scores")

            df["question_fe"] = (
                mkt_adj_weight * df["market_brier_score"]
                + (1 - mkt_adj_weight) * df["question_fe"]
            )

    df["diff_adj_brier_score"] = df["brier_score"] - df["question_fe"]
    return df


def compute_diff_adj_scores(
    df: pd.DataFrame,
    max_model_days_released: int,
    drop_baseline_models: list,
    mkt_adj_weight: float = 1.0,
    exclude_tournament_models_in_2FE: bool = False,
) -> pd.DataFrame:
    """
    Compute difficulty-adjusted Brier scores for all questions, processing
    market and dataset questions separately.

    Args:
        df: DataFrame with forecast data including market_question indicator
        max_model_days_released: Maximum days since model release for inclusion
            in estimation
        drop_baseline_models: List of baseline model names to exclude from estimation
        mkt_adj_weight: If only market questions are present, weight given for market
            Brier score when calculating question difficulties. Question difficulties
            (i.e., question fixed effects) are calculated as
            mkt_adj_weight * market_brier + (1 - mkt_adj_weight) * 2FE_estimate
        exclude_tournament_models_in_2FE: If true, exclude tournament models from the
            estimation of question FE

    Returns:
        DataFrame with difficulty-adjusted Brier scores for both market and
            dataset questions
    """
    df = df.copy()

    # Process market and dataset questions separately
    mask = df["market_question"]

    # Market questions
    df_market = get_diff_adj_brier(
        df=df[mask],
        max_model_days_released=max_model_days_released,
        drop_baseline_models=drop_baseline_models,
        mkt_adj_weight=mkt_adj_weight,
        exclude_tournament_models_in_2FE=exclude_tournament_models_in_2FE,
    )

    # Dataset questions
    df_dataset = get_diff_adj_brier(
        df=df[~mask],
        max_model_days_released=max_model_days_released,
        drop_baseline_models=drop_baseline_models,
        mkt_adj_weight=mkt_adj_weight,
        exclude_tournament_models_in_2FE=exclude_tournament_models_in_2FE,
    )

    # Combine back
    df_combined = pd.concat([df_market, df_dataset], ignore_index=True)

    return df_combined


def create_leaderboard(
    df_with_scores: pd.DataFrame,
    min_days_active_market: int = None,
    min_days_active_dataset: int = None,
) -> pd.DataFrame:
    """
    Create aggregated leaderboard from difficulty-adjusted scores with optional
    activity filters.

    Aggregates model performance across market and dataset questions, applying minimum
    activity thresholds if specified. Models below activity thresholds will show NaN
    for the corresponding metric categories.

    Args:
        df_with_scores: DataFrame with computed difficulty-adjusted scores
        min_days_active_market: Minimum days active required to include market
            scores (optional)
        min_days_active_dataset: Minimum days active required to include dataset
            scores (optional)

    Returns:
        DataFrame with aggregated scores, forecast counts, confidence intervals, and
            activity metrics
    """
    df = df_with_scores.copy()

    # Create base dataframe
    df_res = (
        df[
            [
                "model",
                "model_days_active_market",
                "model_days_active_dataset",
                "model_organization",
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
                df_res[["model", f"model_days_active_{name}", "model_organization"]],
                on="model",
                how="left",
            )

            # Activity filter not specified for ForecastBench models
            # since they do not have a release date
            mask_active = (df_agg[f"model_days_active_{name}"] >= min_days) | (
                df_agg["model_organization"] == "ForecastBench"
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


def _save_and_plot_results(
    df_results: pd.DataFrame,
    name: str,
    results_folder: str,
    metrics: list,
    viz_config: dict,
    metric_labels: dict,
    output_suffix: str = "",
) -> None:
    """Save and plot with configuration support"""

    # Save results with suffix to avoid conflicts
    suffix = f"_{output_suffix}" if output_suffix else ""
    csv_path = f"{results_folder}/stability_{name}{suffix}.csv"
    df_results.to_csv(csv_path, index=False)

    # Create graphs
    if len([m for m in metrics if m in df_results.columns]) > 1:
        n_metrics = len([m for m in metrics if m in df_results.columns])
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        figsize = viz_config.get("figsize", (12, 10))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        plot_idx = 0
        for metric in metrics:
            if metric in df_results.columns and plot_idx < len(axes):
                axes[plot_idx].plot(df_results["x"], df_results[metric], linewidth=2)
                axes[plot_idx].set_xlabel("Days (X in 0-X range)")
                axes[plot_idx].set_ylabel(
                    metric_labels.get(metric, metric.replace("_", " ").title())
                )
                axes[plot_idx].set_title(
                    f"{metric_labels.get(metric, metric)} - {name.title()}"
                )
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_path = f"{results_folder}/stability_{name}{suffix}.png"
        plt.savefig(plot_path, dpi=viz_config.get("dpi", 300))

        if viz_config.get("show_plots", False):
            plt.show()
        else:
            plt.close()


def perform_stability_analysis(
    df_with_scores: pd.DataFrame,
    model_days_active_threshold: int,
    results_folder: str,
    metrics: list = None,
    viz_config: dict = None,
    metric_labels: dict = None,
    output_suffix: str = "",
) -> None:
    """
    Enhanced stability analysis with configurable metrics

    Args:
        df_with_scores: DataFrame with diff-adj scores
        model_days_active_threshold: Min days active for model inclusion
        results_folder: Output directory
        metrics: List of metrics to calculate (defaults to all)
        viz_config: Visualization configuration dict
        output_suffix: Suffix for output filenames to avoid conflicts
    """

    # Default configurations
    if metrics is None:
        metrics = [
            "correlation",
            "rank_correlation",
            "median_displacement",
            "top25_retention",
        ]

    if viz_config is None:
        viz_config = {
            "save_individual_plots": True,
            "save_combined_plot": True,
            "show_plots": False,
            "dpi": 300,
            "figsize": (12, 10),
        }

    if metric_labels is None:
        metric_labels = {
            "correlation": "Score Correlation",
            "rank_correlation": "Rank Correlation (Spearman ρ)",
            "median_displacement": "Median Rank Displacement",
            "top25_retention": "Top-25% Retention Rate",
        }

    df = df_with_scores.copy()
    # Calculate days active at resolution date
    df["days_model_active_at_resolution"] = (
        pd.to_datetime(df["resolution_date"])
        - pd.to_datetime(df["model_first_forecast_date"])
    ).dt.days

    # For comparability, only consider models
    # that are active for at least model_days_active_threshold days
    mask = df["model_days_active"] >= model_days_active_threshold
    df = df[mask].copy()

    # Perform the stability analysis separately for
    # market and dataset questions
    for name, market_mask_val in [
        ("market", True),
        ("dataset", False),
    ]:
        # Calculate the baseline (full period scores)
        mask = df["market_question"] == market_mask_val
        df_full = df[mask].copy()
        df_full_scores = (
            df_full.groupby("model")["diff_adj_brier_score"].mean().reset_index()
        )

        # Calculate all metrics for each time window
        x_values = range(0, model_days_active_threshold + 1)
        results = []

        for x in x_values:
            mask = (
                (df["market_question"] == market_mask_val)
                & (df["days_model_active_at_resolution"] >= 0)
                & (df["days_model_active_at_resolution"] <= x)
            )
            df_incomplete = df[mask].copy()
            df_incomplete_scores = (
                df_incomplete.groupby("model")["diff_adj_brier_score"]
                .mean()
                .reset_index()
            )

            # Calculate metrics using StabilityMetrics class
            if len(df_incomplete_scores) > 0:  # Only if we have data
                metrics_calc = StabilityMetrics(df_full_scores, df_incomplete_scores)

                row = {"x": x}
                if "correlation" in metrics:
                    row["correlation"] = metrics_calc.score_correlation()
                if "rank_correlation" in metrics:
                    row["rank_correlation"] = metrics_calc.rank_correlation()
                if "median_displacement" in metrics:
                    row["median_displacement"] = metrics_calc.median_displacement()
                if "top25_retention" in metrics:
                    row["top25_retention"] = metrics_calc.top_k_retention(25)

                results.append(row)

        # Save and visualize results
        if results:  # Only if we have results
            df_results = pd.DataFrame(results)
            _save_and_plot_results(
                df_results,
                name,
                results_folder,
                metrics,
                viz_config,
                metric_labels,
                output_suffix,
            )


def perform_sample_size_analysis(
    df_with_scores: pd.DataFrame,
    results_folder: str,
    max_thresholds: int = 250,
    max_days: int = 181,
    output_suffix: str = "",
) -> None:
    """
    Perform sample size analysis to determine how many days are needed
    for 80% of models to reach a certain number of resolved questions.

    Args:
        df_with_scores: DataFrame with diff-adj scores and model information
        results_folder: Output directory for results
        max_thresholds: Maximum number of questions to analyze (default 250)
        max_days: Maximum number of days to analyze (default 181: 0-180 inclusive)
        output_suffix: Suffix for output filenames to avoid conflicts
    """

    df = df_with_scores.copy()

    # Perform analysis separately for market and dataset questions
    for name, market_mask_val in [
        ("market", True),
        ("dataset", False),
    ]:
        # Get unique models and their first forecast dates for this question type
        mask = df["market_question"] == market_mask_val
        df_subset = df[mask].copy()

        # Get unique models and their first forecast dates
        df_model_info = (
            df_subset.groupby("model")["model_first_forecast_date"]
            .first()
            .reset_index()
        )

        # Create result list
        result_rows = []

        for _, row in df_model_info.iterrows():
            model = row["model"]
            start_date = pd.to_datetime(row["model_first_forecast_date"])

            # Get all questions for this model
            mask_model = (df_subset["model"] == model) & (
                df_subset["model_organization"] != "ForecastBench"
            )
            model_questions = df_subset[mask_model].copy()

            # Generate dates from start to start + max_days-1 days
            for days_offset in range(max_days):  # 0 to max_days-1 inclusive
                current_date = start_date + timedelta(days=days_offset)

                # Count questions resolved by this date
                num_resolved = model_questions[
                    pd.to_datetime(model_questions["resolution_date"]) <= current_date
                ]["question_id"].nunique()

                result_rows.append(
                    {
                        "model": model,
                        "date": current_date,
                        "days_since_model_first_forecast_date": days_offset,
                        "num_of_resolved_questions": num_resolved,
                    }
                )

        # Create final dataframe
        result_df = pd.DataFrame(result_rows)

        # Calculate for each threshold, when 80% of models reach it
        thresholds = range(1, max_thresholds)
        days_to_threshold = []

        for threshold in thresholds:
            # For each day, calculate % of models that have >= threshold resolved
            daily_pct = result_df.groupby("days_since_model_first_forecast_date").apply(
                lambda x: (x["num_of_resolved_questions"] >= threshold).mean(),
                include_groups=False,
            )

            # Find first day where >= 80% of models have reached threshold
            days_above_80pct = daily_pct[daily_pct >= 0.8]

            if len(days_above_80pct) > 0:
                days_to_threshold.append(
                    {
                        "num_questions": threshold,
                        "days_until_80pct": days_above_80pct.index[0],
                    }
                )

        # Create dataframe and save results
        if days_to_threshold:
            threshold_df = pd.DataFrame(days_to_threshold)

            # Save CSV with suffix to avoid conflicts
            suffix = f"_{output_suffix}" if output_suffix else ""
            csv_path = f"{results_folder}/sample_size_analysis_{name}{suffix}.csv"
            threshold_df.to_csv(csv_path, index=False)

            # Create and save plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                threshold_df["num_questions"],
                threshold_df["days_until_80pct"],
                marker="o",
            )
            plt.xlabel("Number of Resolved Questions ($X$)")
            plt.ylabel("Days Until 80% of Models Reach $X$")
            plt.title(
                f"Days for 80% of Models to Reach $X$ Resolved {name.title()} Questions"
            )
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = f"{results_folder}/sample_size_analysis_{name}{suffix}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()  # Close to avoid displaying

            print(f"Sample size analysis for {name} questions completed:")
            print(f"  - Data saved: {csv_path}")
            print(f"  - Plot saved: {plot_path}")


def generate_trendline_graph(
    df_with_scores: pd.DataFrame,
    df_leaderboard: pd.DataFrame,
    results_folder: str = "",
    output_suffix: str = "",
) -> None:
    """
    Generate CSV data for the interactive trendline graph visualization.

    Creates a CSV file with model performance data over time, formatted for use
    with the trendline_graph.html interactive visualization. The output includes
    difficulty-adjusted Brier scores, sample sizes, confidence intervals, and
    release dates for all models.

    Args:
        df_with_scores: DataFrame with computed difficulty-adjusted scores
        df_leaderboard: DataFrame with leaderboard data including aggregated scores
        results_folder: Directory to save the trendline CSV file
        output_suffix: Suffix to add to the output filename (e.g., "_baseline")

    Returns:
        None (saves CSV file to results_folder/trendline_graph_{suffix}.csv)
    """
    # Collect all required data
    df_trendline = df_leaderboard[
        [
            "model",
            "diff_adj_brier_score_market",
            "diff_adj_brier_score_dataset",
            "diff_adj_brier_score",
            "n_forecasts_market",
            "n_forecasts_dataset",
        ]
    ].copy()
    df_trendline["n_forecasts_overall"] = (
        df_trendline["n_forecasts_market"] + df_trendline["n_forecasts_dataset"]
    )
    df_trendline.rename(
        columns={"diff_adj_brier_score": "diff_adj_brier_score_overall"}, inplace=True
    )
    df_trendline = pd.merge(
        df_trendline,
        df_with_scores[["model", "model_release_date"]].drop_duplicates(),
        how="left",
        validate="1:1",
    )

    # Shape the data into the required form
    # Melt the diff_adj_brier scores
    df_scores = df_trendline.melt(
        id_vars=["model", "model_release_date"],
        value_vars=[
            "diff_adj_brier_score_market",
            "diff_adj_brier_score_dataset",
            "diff_adj_brier_score_overall",
        ],
        var_name="type",
        value_name="diff_adj_brier",
    )

    # Melt the sample sizes
    df_counts = df_trendline.melt(
        id_vars=["model", "model_release_date"],
        value_vars=["n_forecasts_market", "n_forecasts_dataset", "n_forecasts_overall"],
        var_name="type",
        value_name="sample_size",
    )

    # Extract type from column names
    df_scores["type"] = df_scores["type"].str.extract(r"diff_adj_brier_score_(\w+)")
    df_counts["type"] = df_counts["type"].str.extract(r"n_forecasts_(\w+)")

    # Merge scores and counts
    df_final = pd.merge(
        df_scores, df_counts[["model", "type", "sample_size"]], on=["model", "type"]
    )

    # Add confidence interval columns and rename
    df_final["conf_int_lb"] = np.nan
    df_final["conf_int_ub"] = np.nan
    df_final = df_final.rename(columns={"model_release_date": "release_date"})

    # Reorder columns & sort
    df_final = df_final[
        [
            "model",
            "type",
            "diff_adj_brier",
            "sample_size",
            "conf_int_lb",
            "conf_int_ub",
            "release_date",
        ]
    ]
    df_final = df_final.sort_values(by=["model", "release_date"], ascending=True)

    # Save CSV with suffix to avoid conflicts
    suffix = f"_{output_suffix}" if output_suffix else ""
    csv_path = f"{results_folder}/trendline_graph{suffix}.csv"
    df_final.to_csv(csv_path, index=False)
