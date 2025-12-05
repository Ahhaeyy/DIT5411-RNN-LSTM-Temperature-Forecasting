from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Bidirectional, Dropout, Dense
import tensorflow as tf

# ---------------------------------------------------------
# Baseline Model: SimpleRNN (2 layers, 100 units each)
# ---------------------------------------------------------
def create_rnn_model(window_size):
    model = Sequential([
        Input(shape=(window_size, 1)),
        SimpleRNN(100, activation="tanh", return_sequences=True),
        SimpleRNN(100, activation="tanh"),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ---------------------------------------------------------
# Advanced Model: LSTM (2 layers, 100 units + Dropout)
# ---------------------------------------------------------
def create_lstm_model(window_size):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(100, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation="tanh"),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ---------------------------------------------------------
# Optional Extension: BiLSTM (2 layers, 100 units + Dropout)
# ---------------------------------------------------------
def create_bilstm_model(window_size):
    model = Sequential([
        Input(shape=(window_size, 1)),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(100)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model
