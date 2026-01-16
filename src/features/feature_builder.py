import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from src.sql_feature_queries import build_features_from_sql


# ---------------------------
# Engagement Velocity
# ---------------------------
def compute_engagement_velocity(behavior_df):
    records = []

    for customer_id, group in behavior_df.groupby("customer_id"):
        if group["week"].nunique() < 5:
            continue

        slope, _ = polyfit(group["week"], group["weekly_usage"], 1)
        records.append({
            "customer_id": customer_id,
            "engagement_velocity": slope
        })

    return pd.DataFrame(records)


# ---------------------------
# Average Usage (Level)
# ---------------------------
def compute_avg_usage(behavior_df):
    return (
        behavior_df
        .groupby("customer_id")["weekly_usage"]
        .mean()
        .reset_index(name="avg_weekly_usage")
    )


# ---------------------------
# Friction Signal
# ---------------------------
def compute_total_friction(behavior_df):
    return (
        behavior_df
        .groupby("customer_id")["support_tickets"]
        .sum()
        .reset_index(name="total_support_tickets")
    )


# ---------------------------
# Build Feature Table (SQL-backed)
# ---------------------------
def build_feature_table():
    customers_df, behavior_df = build_features_from_sql()

    avg_usage = compute_avg_usage(behavior_df)
    velocity = compute_engagement_velocity(behavior_df)
    friction = compute_total_friction(behavior_df)

    recency = compute_recency(behavior_df)
    frequency = compute_frequency(behavior_df)
    recent_velocity = compute_recent_velocity(behavior_df)
    friction_intensity = compute_recent_friction_intensity(behavior_df)

    features = (
        customers_df
        .merge(avg_usage, on="customer_id", how="left")
        .merge(velocity, on="customer_id", how="left")
        .merge(friction, on="customer_id", how="left")
        .merge(recency, on="customer_id", how="left")
        .merge(frequency, on="customer_id", how="left")
        .merge(recent_velocity, on="customer_id", how="left")
        .merge(friction_intensity, on="customer_id", how="left")
    )

    return features


# ---------------------------
# Recency (weeks since last usage)
# ---------------------------
def compute_recency(behavior_df, usage_threshold=1.0):
    last_active_week = (
        behavior_df[behavior_df["weekly_usage"] > usage_threshold]
        .groupby("customer_id")["week"]
        .max()
        .reset_index(name="last_active_week")
    )

    max_week = behavior_df["week"].max()
    last_active_week["recency"] = max_week - last_active_week["last_active_week"]

    return last_active_week[["customer_id", "recency"]]


# ---------------------------
# Frequency (active weeks in last N weeks)
# ---------------------------
def compute_frequency(behavior_df, window=12, usage_threshold=1.0):
    max_week = behavior_df["week"].max()
    recent = behavior_df[behavior_df["week"] >= max_week - window]

    freq = (
        recent[recent["weekly_usage"] > usage_threshold]
        .groupby("customer_id")["week"]
        .count()
        .reset_index(name="frequency")
    )

    return freq


# ---------------------------
# Recent Engagement Velocity
# ---------------------------
def compute_recent_velocity(behavior_df, window=8):
    records = []
    max_week = behavior_df["week"].max()
    recent = behavior_df[behavior_df["week"] >= max_week - window]

    for customer_id, group in recent.groupby("customer_id"):
        if group["week"].nunique() < 4:
            continue

        slope, _ = polyfit(group["week"], group["weekly_usage"], 1)
        records.append({
            "customer_id": customer_id,
            "recent_engagement_velocity": slope
        })

    return pd.DataFrame(records)


# ---------------------------
# Recent Friction Intensity
# ---------------------------
def compute_recent_friction_intensity(behavior_df, window=8):
    max_week = behavior_df["week"].max()
    recent = behavior_df[behavior_df["week"] >= max_week - window]

    agg = (
        recent
        .groupby("customer_id")
        .agg(
            recent_support_tickets=("support_tickets", "sum"),
            recent_usage=("weekly_usage", "sum")
        )
        .reset_index()
    )

    agg["friction_intensity"] = (
        agg["recent_support_tickets"] / (agg["recent_usage"] + 1e-6)
    )

    return agg[["customer_id", "friction_intensity"]]
