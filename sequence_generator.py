"""
Sequence Generator for Time-Series Forecasting
DIT5411 Machine Learning Project

This module performs the following:
 - Loads the processed dataset (1980–2025)
 - Splits the data into training (1980–2024) and testing (2025)
 - Generates sliding-window sequences for RNN/LSTM models
 - Normalizes data using MinMaxScaler (fit only on training data)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Project-required date ranges (as stated in the PDF)
# ------------------------------------------------------------
TRAIN_END = pd.to_datetime("2024-12-31")
TEST_START = pd.to_datetime("2025-01-01")
TEST_END = pd.to_datetime("2025-10-30")


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
def load_processed_data(path="processed_HKO_GMT_ALL.csv"):
    """
    Load the cleaned dataset and ensure it is chronologically sorted.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    return df


# ------------------------------------------------------------
# Sliding-window sequence generator
# ------------------------------------------------------------
def create_sequences(values, window_size):
    """
    Convert a 1D sequence into sliding-window samples.

    Args:
        values (array): Normalized values, shape [N, 1]
        window_size (int): Number of past days used as input

    Returns:
        X (array): shape [samples, window_size, 1]
        y (array): shape [samples, 1]
    """
    X, y = [], []

    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])

    X, y = np.array(X), np.array(y)

    return X, y


# ------------------------------------------------------------
# Train-test split & sequence preparation
# ------------------------------------------------------------
def get_train_test_data(window_size=45, path="processed_HKO_GMT_ALL.csv"):
    """
    Load dataset, normalize values, and generate sequences for training and testing.

    Training: 1980–2024
    Testing : 2025-01-01 to 2025-10-30

    Returns:
        X_train, y_train, X_test, y_test : arrays ready for model training
        scaler : fitted MinMaxScaler for inverse-transform
        test_dates : dates aligned with y_test for plotting
    """

    df = load_processed_data(path)

    # --------------------------
    # Split by date
    # --------------------------
    df_train = df[df["date"] <= TRAIN_END]
    df_test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]

    # --------------------------
    # Fit scaler on training only
    # --------------------------
    scaler = MinMaxScaler()
    train_values = df_train["Value"].values.reshape(-1, 1)
    test_values = df_test["Value"].values.reshape(-1, 1)

    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    # --------------------------
    # Generate sliding-window sequences
    # --------------------------
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    # Dates corresponding to each prediction point in test set
    test_dates = df_test["date"].values[window_size:]

    # Basic validation
    assert len(X_test) == len(y_test) == len(test_dates), "Test alignment mismatch"

    return X_train, y_train, X_test, y_test, scaler, test_dates


# ------------------------------------------------------------
# Allow standalone execution (debug mode)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Testing sequence generator...")

    X_train, y_train, X_test, y_test, _, test_dates = get_train_test_data(45)

    print("Train samples:", X_train.shape)
    print("Test samples:", X_test.shape)
    print("Example test date:", test_dates[0])
