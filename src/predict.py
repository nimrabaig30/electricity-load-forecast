import os
import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

    # Load historical electricity data
    df = load_raw_data(RAW_DATA_PATH)

    # Create total load and features
    df = create_total_load(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Prepare the modeling dataset
    model_df = prepare_model_data(df)

    # Use the same chronological 80/20 split as training
    split_index = int(len(model_df) * 0.8)
    test_df = model_df.iloc[split_index:].copy()

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

    X_test = test_df[features]
    y_test = test_df["total_load"]

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Predict electricity load for the test period
    predictions = model.predict(X_test)

    # Compare actual and predicted values
    results = pd.DataFrame({
        "timestamp": test_df["timestamp"],
        "actual_load": y_test,
        "predicted_load": predictions
    })

    results["absolute_error"] = (
        results["actual_load"] - results["predicted_load"]
    ).abs()

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        results["actual_load"],
        results["predicted_load"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            results["actual_load"],
            results["predicted_load"]
        )
    )

    # Save predictions
    os.makedirs("outputs", exist_ok=True)

    results.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False
    )

    # Display results
    print("\nSample Predictions:")
    print(results.head())

    print("\nTest Set Evaluation:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    print("\nPrediction completed!")
    print("Rows predicted:", len(results))
    print("Predictions saved to:", PREDICTIONS_OUTPUT_PATH)


if __name__ == "__main__":
    generate_predictions()