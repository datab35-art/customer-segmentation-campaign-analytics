"""K-Means segmentation with evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(X, k_range=range(2, 10)):
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=min(5000, len(X))))

    return list(k_range), inertias, silhouettes


def fit_kmeans(X, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    return km, labels, sil


def profile_segments(df: pd.DataFrame, features: list) -> pd.DataFrame:
    profiles = df.groupby("segment")[features].agg(["mean", "median", "std"])
    profiles.columns = ["_".join(col) for col in profiles.columns]
    profiles["segment_size"] = df.groupby("segment").size()
    profiles["segment_pct"] = (profiles["segment_size"] / len(df) * 100).round(1)
    return profiles


SEGMENT_LABELS = {
    "high_engagement_high_cost": "High-Value Engaged",
    "low_engagement_at_risk": "Cost-Sensitive At-Risk",
    "young_digital": "Young Digital-First",
    "stable_low_touch": "Stable Low-Touch",
}


def label_segments(df: pd.DataFrame, features_used: list) -> pd.DataFrame:
    """Assign interpretable labels based on segment centroids."""
    df = df.copy()
    seg_profiles = df.groupby("segment").agg(
        avg_age=("age", "mean"),
        avg_digital=("digital_engagement", "mean"),
        avg_cost=("total_cost_ytd", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
    )

    label_map = {}
    sorted_by_digital = seg_profiles.sort_values("avg_digital", ascending=False)

    assigned = set()
    # youngest + most digital → Young Digital-First
    youngest = seg_profiles.sort_values("avg_age").index[0]
    label_map[youngest] = "Young Digital-First"
    assigned.add(youngest)

    # highest cost + engagement → High-Value Engaged
    remaining = seg_profiles.drop(index=list(assigned))
    high_val = remaining.sort_values("avg_cost", ascending=False).index[0]
    label_map[high_val] = "High-Value Engaged"
    assigned.add(high_val)

    # lowest satisfaction → Cost-Sensitive At-Risk
    remaining = seg_profiles.drop(index=list(assigned))
    at_risk = remaining.sort_values("avg_satisfaction").index[0]
    label_map[at_risk] = "Cost-Sensitive At-Risk"
    assigned.add(at_risk)

    # remainder → Stable Low-Touch
    for seg in seg_profiles.index:
        if seg not in assigned:
            label_map[seg] = "Stable Low-Touch"

    df["segment_label"] = df["segment"].map(label_map)
    return df, label_map
