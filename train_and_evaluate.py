# Trains RNN and LSTM models, evaluates MAE/RMSE, saves plots and models.

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sequence_generator import get_train_test_data, WINDOW_SIZE
from models_rnn_lstm import create_rnn_model, create_lstm_model

def main():
    X_train, y_train, X_test, y_test, scaler, test_dates = get_train_test_data()
    rnn_model = create_rnn_model(WINDOW_SIZE)
    lstm_model = create_lstm_model(WINDOW_SIZE)

    rnn_model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1)
    lstm_model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1)

    y_pred_rnn = rnn_model.predict(X_test)
    y_pred_lstm = lstm_model.predict(X_test)

    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_rnn_inv = scaler.inverse_transform(y_pred_rnn)
    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)

    mae_rnn = mean_absolute_error(y_test_inv, y_pred_rnn_inv)
    rmse_rnn = math.sqrt(mean_squared_error(y_test_inv, y_pred_rnn_inv))
    mae_lstm = mean_absolute_error(y_test_inv, y_pred_lstm_inv)
    rmse_lstm = math.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))

    print("RNN MAE:", mae_rnn)
    print("RNN RMSE:", rmse_rnn)
    print("LSTM MAE:", mae_lstm)
    print("LSTM RMSE:", rmse_lstm)

    os.makedirs("models", exist_ok=True)
    rnn_model.save("models/rnn_model.h5")
    lstm_model.save("models/lstm_model.h5")

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12,5))
    plt.plot(test_dates, y_test_inv, label="Actual")
    plt.plot(test_dates, y_pred_rnn_inv, label="RNN")
    plt.plot(test_dates, y_pred_lstm_inv, label="LSTM")
    plt.title("Actual vs Predicted Grass Minimum Temperature")
    plt.legend()
    plt.savefig("figures/actual_vs_predicted.png")
    plt.close()

    residuals_rnn = y_test_inv - y_pred_rnn_inv
    residuals_lstm = y_test_inv - y_pred_lstm_inv

    plt.figure(figsize=(8,5))
    plt.hist(residuals_rnn, bins=30, alpha=0.6, label="RNN")
    plt.hist(residuals_lstm, bins=30, alpha=0.6, label="LSTM")
    plt.title("Error Distribution")
    plt.legend()
    plt.savefig("figures/error_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()
