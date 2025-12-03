"""
Model Definitions for DIT5411 Machine Learning Project

This module defines:
 - Simple RNN model (baseline)
 - LSTM model (advanced)
 - Bidirectional LSTM model (optional extension)

All models use:
 - Sequence-to-one prediction
 - Adam optimizer
 - MSE loss
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Bidirectional, Dropout, Dense


# ------------------------------------------------------------
# Baseline RNN
# ------------------------------------------------------------
def create_rnn_model(window_size):
    """
    Create a simple RNN model (baseline model).
    """
    model = Sequential([
        Input(shape=(window_size, 1)),
        SimpleRNN(64, activation="tanh"),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model


# ------------------------------------------------------------
# LSTM (advanced model)
# ------------------------------------------------------------
def create_lstm_model(window_size):
    """
    Create an LSTM model with dropout.
    """
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(64, activation="tanh"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model


# ------------------------------------------------------------
# Optional Extension: Bidirectional LSTM
# ------------------------------------------------------------
def create_bilstm_model(window_size):
    """
    Create a Bidirectional LSTM model.
    """
    model = Sequential([
        Input(shape=(window_size, 1)),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model
