Electricity Load Forecasting

A machine learning project for forecasting electricity demand using historical electricity consumption data.

Project Overview

This project predicts electricity load using the UCI ElectricityLoadDiagrams20112014 dataset.

The dataset contains electricity consumption data from 370 clients recorded at 15-minute intervals.

The project includes:

Data validation
Feature engineering
Time-series forecasting
Random Forest regression
Model evaluation
Prediction generation
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Jupyter Notebook
Features Used
Time Features
Hour
Day of week
Day of year
Month
Weekend indicator
Lag Features
lag_1 — previous 15-minute load
lag_4 — previous 1-hour load
lag_96 — previous 24-hour load
Rolling Features
4-period rolling mean
96-period rolling mean
96-period rolling standard deviation

Rolling features use only past observations to avoid data leakage.

Model

The final model is a Random Forest Regressor.

Parameters:

n_estimators: 100
random_state: 42
n_jobs: -1

The data is divided chronologically into 80% training data and 20% testing data.

Results
Model	MAE	RMSE
Previous-Day Baseline	9043.20	14431.27
Random Forest	4419.17	6743.08
Improved Random Forest	4230.12	6451.62

The Improved Random Forest achieved:

53.22% lower MAE than the baseline
55.29% lower RMSE than the baseline
Data Validation

Before training, the pipeline checks:

Missing values
Duplicate timestamps
Timestamp ordering
Required timestamp column
Electricity consumption columns
Project Structure

electricity_load_forecast/

├── data/

│ ├── raw/

│ └── processed/

├── notebooks/

│ └── 01_data_inspection.ipynb

├── src/

│ ├── data_processing.py

│ ├── validate_data.py

│ ├── train_model.py

│ └── predict.py

├── outputs/

├── README.md

├── requirements.txt

└── .gitignore

How to Run

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Place the dataset file:

LD2011_2014.txt

inside:

data/raw/

Train the model:

python src/train_model.py

Generate predictions:

python src/predict.py

Dataset

ElectricityLoadDiagrams20112014

Source: UCI Machine Learning Repository

The raw dataset is not included in this repository because of its large size.

Future Improvements
Hyperparameter tuning
XGBoost comparison
Multi-step forecasting
Real-time prediction
Model monitoring
REST API
Streamlit dashboard
Author

Nimra Baig
Computer Science Engineering — Data Science