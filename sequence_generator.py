# Converts the cleaned dataset into 45-day sliding window sequences.
# Produces X_train, y_train, X_test, y_test for RNN/LSTM training.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 45
TRAIN_END_DATE = pd.to_datetime("2024-12-31")
TEST_START_DATE = pd.to_datetime("2025-01-01")
TEST_END_DATE = pd.to_datetime("2025-10-30")

def load_processed_data(path="processed_HKO_GMT_ALL.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    return df

def create_sequences(values, window_size):
    X = []
    y = []
    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

def get_train_test_data(window_size=WINDOW_SIZE, path="processed_HKO_GMT_ALL.csv"):
    df = load_processed_data(path)
    df = df[(df["date"] >= pd.to_datetime("1980-01-01")) & (df["date"] <= TEST_END_DATE)]
    df_train = df[df["date"] <= TRAIN_END_DATE]
    df_test = df[(df["date"] >= TEST_START_DATE) & (df["date"] <= TEST_END_DATE)]
    scaler = MinMaxScaler()
    train_values = df_train["Value"].values.reshape(-1, 1)
    test_values = df_test["Value"].values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)
    test_dates = df_test["date"].values[window_size:]
    return X_train, y_train, X_test, y_test, scaler, test_dates
