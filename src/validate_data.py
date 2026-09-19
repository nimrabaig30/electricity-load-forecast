import pandas as pd


def validate_data(df):
    """Validate the raw electricity consumption dataset."""

    print("Starting data validation...")

    # Check required timestamp column
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    # Check for duplicate timestamps
    duplicate_timestamps = df["timestamp"].duplicated().sum()

    if duplicate_timestamps > 0:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps."
        )

    # Check for missing values
    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    # Check timestamp ordering
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in chronological order.")

    # Check electricity consumption columns
    consumption_columns = [
        col for col in df.columns
        if col.startswith("MT_")
    ]

    if len(consumption_columns) == 0:
        raise ValueError("No electricity consumption columns found.")

    print("Data validation passed!")
    print("Rows:", len(df))
    print("Consumption columns:", len(consumption_columns))
    print("Missing values:", missing_values)
    print("Duplicate timestamps:", duplicate_timestamps)

    return True