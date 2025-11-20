# Defines two deep learning models:
# 1. Simple RNN
# 2. LSTM with dropout regularization

import tensorflow as tf

def create_rnn_model(window_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(64, activation="tanh", input_shape=(window_size, 1)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model

def create_lstm_model(window_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, activation="tanh", input_shape=(window_size, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model
