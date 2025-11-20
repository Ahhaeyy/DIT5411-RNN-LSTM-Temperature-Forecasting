import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sequence_generator import create_sequences
from models_rnn_lstm import create_rnn_model

df = pd.read_csv("processed_HKO_GMT_ALL.csv")
values = df["Value"].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

sequence_lengths = [30, 45, 60]

for n_steps in sequence_lengths:
    print(f"\nTesting sequence length = {n_steps}")

    X, y = create_sequences(scaled, n_steps)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = create_rnn_model(n_steps)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    pred = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred)
    y_test_inv = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_inv, pred_inv)
    print("MAE:", mae)
