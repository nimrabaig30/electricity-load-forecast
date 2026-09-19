import os
import joblib
import pandas as pd

from data_processing import (
    load_raw_data,
    create_total_load,
    create_time_features,
    create_lag_features,
    create_rolling_features,
    prepare_model_data
)


RAW_DATA_PATH = "data/raw/LD2011_2014.txt"
MODEL_PATH = "outputs/electricity_load_model.pkl"
PREDICTIONS_OUTPUT_PATH = "outputs/electricity_load_predictions.csv"


def generate_predictions():

    # Load raw data
    df = load_raw_data(RAW_DATA_PATH)

    # Create features
    df = create_total_load(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Prepare model data
    model_df = prepare_model_data(df)

    # Features used by the trained model
    features = [
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

    X = model_df[features]

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Generate predictions
    predictions = model.predict(X)

    # Create prediction results
    results = pd.DataFrame({
        "timestamp": model_df["timestamp"],
        "actual_load": model_df["total_load"],
        "predicted_load": predictions
    })

    # Calculate absolute error
    results["absolute_error"] = (
        results["actual_load"] - results["predicted_load"]
    ).abs()

    # Create output directory if needed
    os.makedirs("outputs", exist_ok=True)

    # Save predictions
    results.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False
    )

    # Display sample predictions
    print("\nSample predictions:")
    print(results.head())

    print("\nPrediction completed!")
    print("Rows predicted:", len(results))
    print("Predictions saved to:", PREDICTIONS_OUTPUT_PATH)


if __name__ == "__main__":
    generate_predictions()