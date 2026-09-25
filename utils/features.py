"""
Data validation + feature engineering shared by Upload and Feature Engineering pages.
"""
import pandas as pd

# Columns the uploaded historical CSV MUST contain.
REQUIRED_COLUMNS = [
    "customer_id",
    "service_type",
    "booking_date",
    "appointment_date",
    "appointment_time",
    "previous_appointments",
    "previous_noshows",
    "reminder_sent",
    "no_show",
]

# Final feature set used for model training / prediction.
FEATURE_COLUMNS = [
    "lead_time_days",
    "appointment_hour",
    "is_weekend",
    "previous_appointments",
    "previous_noshows",
    "previous_attendance_rate",
    "reminder_sent",
    "service_type",   # categorical -> one-hot encoded at train time
    "day_of_week",    # categorical -> one-hot encoded at train time
]


def validate_columns(df: pd.DataFrame):
    """Returns (is_valid, missing_columns list)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0), missing


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates, handle obvious missing values."""
    df = df.drop_duplicates()

    # Coerce reminder_sent / no_show to 0/1 ints regardless of True/False/Yes/No input
    for col in ["reminder_sent", "no_show"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
                .fillna(0)
                .astype(int)
            )

    # Fill missing numeric history fields with 0 (new customer assumption)
    for col in ["previous_appointments", "previous_noshows"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Drop rows where core dates are unparseable
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
    df = df.dropna(subset=["booking_date", "appointment_date"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived columns needed for EDA + modeling:
    lead_time_days, appointment_hour, day_of_week, is_weekend, previous_attendance_rate
    """
    df = df.copy()

    df["lead_time_days"] = (df["appointment_date"] - df["booking_date"]).dt.days.clip(lower=0)

    # appointment_time expected like "14:30" or "2:30 PM" - try common formats first, then fall back
    time_str = df["appointment_time"].astype(str).str.strip()
    parsed_time = pd.to_datetime(time_str, format="%H:%M", errors="coerce")
    still_missing = parsed_time.isna()
    if still_missing.any():
        parsed_time.loc[still_missing] = pd.to_datetime(
            time_str[still_missing], format="%I:%M %p", errors="coerce"
        )
    still_missing = parsed_time.isna()
    if still_missing.any():
        parsed_time.loc[still_missing] = pd.to_datetime(time_str[still_missing], errors="coerce")
    df["appointment_hour"] = parsed_time.dt.hour.fillna(12).astype(int)

    df["day_of_week"] = df["appointment_date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    total_prior = df["previous_appointments"].replace(0, pd.NA)
    df["previous_attendance_rate"] = (
        1 - (df["previous_noshows"] / total_prior)
    ).fillna(1.0).clip(0, 1)

    return df


def risk_level(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"
