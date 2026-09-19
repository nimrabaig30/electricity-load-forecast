import pandas as pd


def load_raw_data(file_path):
    """Load the raw electricity consumption dataset."""

    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=","
    )

    df.rename(
        columns={"Unnamed: 0": "timestamp"},
        inplace=True
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Defragment the DataFrame after loading
    df = df.copy()

    return df


def create_total_load(df):
    """Calculate total electricity load across all clients."""

    consumption_columns = [
        col for col in df.columns
        if col.startswith("MT_")
    ]

    total_load = df[consumption_columns].sum(axis=1)

    df = pd.concat(
        [
            df,
            total_load.rename("total_load")
        ],
        axis=1
    )

    return df


def create_time_features(df):
    """Create calendar-based time features."""

    time_features = pd.DataFrame({
        "hour": df["timestamp"].dt.hour,
        "day_of_week": df["timestamp"].dt.dayofweek,
        "day_of_year": df["timestamp"].dt.dayofyear,
        "month": df["timestamp"].dt.month,
        "year": df["timestamp"].dt.year,
    })

    time_features["is_weekend"] = (
        time_features["day_of_week"] >= 5
    ).astype(int)

    df = pd.concat(
        [df, time_features],
        axis=1
    )

    return df


def create_lag_features(df):
    """Create historical electricity load features."""

    lag_features = pd.DataFrame({
        "lag_1": df["total_load"].shift(1),
        "lag_4": df["total_load"].shift(4),
        "lag_96": df["total_load"].shift(96),
    })

    df = pd.concat(
        [df, lag_features],
        axis=1
    )

    return df


def create_rolling_features(df):
    """Create rolling statistics using only past observations."""

    rolling_features = pd.DataFrame({
        "rolling_mean_4": (
            df["total_load"]
            .shift(1)
            .rolling(4)
            .mean()
        ),

        "rolling_mean_96": (
            df["total_load"]
            .shift(1)
            .rolling(96)
            .mean()
        ),

        "rolling_std_96": (
            df["total_load"]
            .shift(1)
            .rolling(96)
            .std()
        ),
    })

    df = pd.concat(
        [df, rolling_features],
        axis=1
    )

    return df


def prepare_model_data(df):
    """Prepare the final dataset for machine learning."""

    model_columns = [
        "timestamp",
        "total_load",
        "hour",
        "day_of_week",
        "day_of_year",
        "month",
        "is_weekend",
        "lag_1",
        "lag_4",
        "lag_96",
        "rolling_mean_4",
        "rolling_mean_96",
        "rolling_std_96"
    ]

    model_df = df[model_columns].copy()

    model_df = (
        model_df
        .dropna()
        .reset_index(drop=True)
    )

    return model_df