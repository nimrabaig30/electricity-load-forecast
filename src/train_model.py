import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_processing import (
    load_raw_data,
    create_total_load,
    create_time_features,
    create_lag_features,
    create_rolling_features,
    prepare_model_data
)

from validate_data import validate_data


RAW_DATA_PATH = "data/raw/LD2011_2014.txt"
MODEL_OUTPUT_PATH = "outputs/electricity_load_model.pkl"
EVALUATION_OUTPUT_PATH = "outputs/model_evaluation.csv"


def train_model():

    # Load and process data
    df = load_raw_data(RAW_DATA_PATH)

    validate_data(df)

    df = create_total_load(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    model_df = prepare_model_data(df)

    # Chronological train/test split
    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index]
    test_df = model_df.iloc[split_index:]

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

    X_train = train_df[features]
    y_train = train_df["total_load"]

    X_test = test_df[features]
    y_test = test_df["total_load"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Save model
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)

    # Save evaluation
    evaluation = pd.DataFrame({
        "Model": ["Improved Random Forest"],
        "MAE": [mae],
        "RMSE": [rmse]
    })

    evaluation.to_csv(
        EVALUATION_OUTPUT_PATH,
        index=False
    )

    print("Model training completed!")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("Model saved to:", MODEL_OUTPUT_PATH)
    print("Evaluation saved to:", EVALUATION_OUTPUT_PATH)


if __name__ == "__main__":
    train_model()