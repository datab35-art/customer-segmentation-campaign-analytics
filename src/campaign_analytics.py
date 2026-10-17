"""Campaign performance analytics by segment."""

import pandas as pd
import numpy as np


def campaign_summary_by_segment(campaigns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    merged = campaigns.merge(members[["member_id", "segment_label"]], on="member_id", how="left")

    summary = merged.groupby("segment_label").agg(
        total_touches=("campaign_id", "count"),
        open_rate=("opened", "mean"),
        click_rate=("clicked", "mean"),
        conversion_rate=("converted", "mean"),
        total_cost=("cost", "sum"),
        conversions=("converted", "sum"),
    ).round(4)

    summary["cost_per_conversion"] = np.where(
        summary["conversions"] > 0,
        (summary["total_cost"] / summary["conversions"]).round(2),
        np.nan,
    )
    return summary


def campaign_summary_by_channel(campaigns: pd.DataFrame) -> pd.DataFrame:
    summary = campaigns.groupby("channel").agg(
        total_touches=("campaign_id", "count"),
        open_rate=("opened", "mean"),
        click_rate=("clicked", "mean"),
        conversion_rate=("converted", "mean"),
        total_cost=("cost", "sum"),
        conversions=("converted", "sum"),
    ).round(4)

    summary["cost_per_conversion"] = np.where(
        summary["conversions"] > 0,
        (summary["total_cost"] / summary["conversions"]).round(2),
        np.nan,
    )
    return summary


def campaign_summary_by_type(campaigns: pd.DataFrame) -> pd.DataFrame:
    summary = campaigns.groupby("campaign_type").agg(
        total_touches=("campaign_id", "count"),
        open_rate=("opened", "mean"),
        click_rate=("clicked", "mean"),
        conversion_rate=("converted", "mean"),
        total_cost=("cost", "sum"),
        conversions=("converted", "sum"),
    ).round(4)

    summary["cost_per_conversion"] = np.where(
        summary["conversions"] > 0,
        (summary["total_cost"] / summary["conversions"]).round(2),
        np.nan,
    )
    return summary


def segment_channel_heatmap_data(campaigns: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    merged = campaigns.merge(members[["member_id", "segment_label"]], on="member_id", how="left")
    pivot = merged.pivot_table(
        index="segment_label",
        columns="channel",
        values="conversion_rate" if "conversion_rate" in merged.columns else "converted",
        aggfunc="mean",
    ).round(4)
    return pivot
