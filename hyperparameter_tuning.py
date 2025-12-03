"""
DIT5411 Machine Learning Project  
Hyperparameter Tuning for Sliding Window Length

This script evaluates different sequence lengths (window sizes)
to determine which provides the best forecasting performance.
It follows the project PDF requirement:
 - Training data: 1980–2024
 - Testing data: 2025-01-01 to 2025-10-30
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sequence_generator import get_train_test_data
from models_rnn_lstm import create_rnn_model


def tune_sequence_length():
    """
    Test different sliding window sizes using date-based splitting.
    Only the RNN model is used here for simplicity, as required by the project.
    """

    window_sizes = [30, 45, 60]  # Hyperparameter candidates

    for window_size in window_sizes:
        print(f"\n=== Testing window size: {window_size} days ===")

        # Load train/test sets according to project PDF rules
        X_train, y_train, X_test, y_test, scaler, test_dates = get_train_test_data(window_size)

        # Build RNN model
        model = create_rnn_model(window_size)

        # Train model
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        # Predict on test set
        preds = model.predict(X_test)
        preds_inv = scaler.inverse_transform(preds)
        y_test_inv = scaler.inverse_transform(y_test)

        # Compute MAE
        mae = mean_absolute_error(y_test_inv, preds_inv)
        print(f"MAE for window size {window_size}: {mae:.4f}")


if __name__ == "__main__":
    tune_sequence_length()
