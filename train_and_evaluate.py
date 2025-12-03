import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sequence_generator import get_train_test_data
from models_rnn_lstm import create_rnn_model, create_lstm_model, create_bilstm_model


def main():

    # ======================================================
    # Load dataset according to PDF requirement:
    # Train: 1980–2024
    # Test: 2025-01-01 to 2025-10-30
    # ======================================================
    window_size = 45
    X_train, y_train, X_test, y_test, scaler, test_dates = get_train_test_data(window_size)

    # Inverse-scale true labels
    y_test_inv = scaler.inverse_transform(y_test)

    # ======================================================
    # 1. Train RNN
    # ======================================================
    print("\nTraining RNN...")
    rnn = create_rnn_model(window_size)
    rnn.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    y_pred_rnn = rnn.predict(X_test)
    y_pred_rnn_inv = scaler.inverse_transform(y_pred_rnn)
    rnn.save("models/rnn_model.h5")

    # ======================================================
    # 2. Train LSTM
    # ======================================================
    print("\nTraining LSTM...")
    lstm = create_lstm_model(window_size)
    lstm.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    y_pred_lstm = lstm.predict(X_test)
    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
    lstm.save("models/lstm_model.h5")

    # ======================================================
    # 3. Train Bidirectional LSTM
    # ======================================================
    print("\nTraining BiLSTM...")
    bilstm = create_bilstm_model(window_size)
    bilstm.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    y_pred_bilstm = bilstm.predict(X_test)
    y_pred_bilstm_inv = scaler.inverse_transform(y_pred_bilstm)
    bilstm.save("models/bilstm_model.h5")

    # ======================================================
    # Evaluation
    # ======================================================
    print("\nEvaluating models...\n")

    rnn_mae = mean_absolute_error(y_test_inv, y_pred_rnn_inv)
    rnn_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_rnn_inv))

    lstm_mae = mean_absolute_error(y_test_inv, y_pred_lstm_inv)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))

    bilstm_mae = mean_absolute_error(y_test_inv, y_pred_bilstm_inv)
    bilstm_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_bilstm_inv))

    print("RNN MAE:", rnn_mae)
    print("RNN RMSE:", rnn_rmse)
    print("LSTM MAE:", lstm_mae)
    print("LSTM RMSE:", lstm_rmse)
    print("BiLSTM MAE:", bilstm_mae)
    print("BiLSTM RMSE:", bilstm_rmse)

    # ======================================================
    # Plot Actual vs Predicted
    # ======================================================
    plt.figure(figsize=(14, 5))
    plt.plot(test_dates, y_test_inv, label="Actual")
    plt.plot(test_dates, y_pred_rnn_inv, label="RNN")
    plt.plot(test_dates, y_pred_lstm_inv, label="LSTM")
    plt.plot(test_dates, y_pred_bilstm_inv, label="BiLSTM")

    plt.title("Actual vs Predicted Grass Minimum Temperature (2025 Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/actual_vs_predicted.png")
    plt.close()

    # ======================================================
    # Plot Error Distribution
    # ======================================================
    plt.figure(figsize=(8, 5))
    plt.hist(y_test_inv - y_pred_rnn_inv, bins=40, alpha=0.5, label="RNN")
    plt.hist(y_test_inv - y_pred_lstm_inv, bins=40, alpha=0.5, label="LSTM")
    plt.hist(y_test_inv - y_pred_bilstm_inv, bins=40, alpha=0.5, label="BiLSTM")

    plt.title("Prediction Error Distribution (2025 Test Set)")
    plt.xlabel("Error (°C)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/error_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()
