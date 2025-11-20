# This file defines three deep learning models:
# 1. Simple RNN
# 2. LSTM with dropout
# 3. Bidirectional LSTM (optional advanced model)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional, Dropout, Dense

def create_rnn_model(window_size):
    model = Sequential()
    model.add(SimpleRNN(64, activation="tanh", input_shape=(window_size, 1)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model

def create_lstm_model(window_size):
    model = Sequential()
    model.add(LSTM(64, activation="tanh", input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model

def create_bilstm_model(window_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False),
                            input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model
