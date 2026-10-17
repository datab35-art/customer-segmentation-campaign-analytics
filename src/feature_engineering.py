"""Feature engineering for member segmentation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


SEGMENTATION_FEATURES = [
    "age",
    "tenure_months",
    "annual_claims",
    "er_visits",
    "rx_fills",
    "preventive_visits",
    "portal_logins_6m",
    "mobile_sessions_6m",
    "calls_to_service_6m",
    "email_opens_6m",
    "total_cost_ytd",
    "member_cost_share_ytd",
    "satisfaction_score",
]


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["digital_engagement"] = df["portal_logins_6m"] + df["mobile_sessions_6m"]
    df["cost_ratio"] = np.where(
        df["total_cost_ytd"] > 0,
        df["member_cost_share_ytd"] / df["total_cost_ytd"],
        0,
    )
    df["utilization_intensity"] = df["annual_claims"] + df["er_visits"] + df["rx_fills"]
    df["engagement_per_claim"] = np.where(
        df["annual_claims"] > 0,
        df["digital_engagement"] / df["annual_claims"],
        df["digital_engagement"],
    )
    return df


EXTENDED_FEATURES = SEGMENTATION_FEATURES + [
    "digital_engagement",
    "cost_ratio",
    "utilization_intensity",
    "engagement_per_claim",
]


def scale_features(df: pd.DataFrame, features: list = None) -> tuple:
    features = features or EXTENDED_FEATURES
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return X, scaler, features
