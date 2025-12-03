# DIT5411 Machine Learning Project
## Forecasting Hong Kong Daily Grass Minimum Temperature using RNN, LSTM, and BiLSTM

This project develops and compares three deep learning models—Simple RNN, LSTM, and Bidirectional LSTM—to forecast Hong Kong’s daily grass minimum temperature using historical data from the Hong Kong Observatory (HKO). The workflow follows the official DIT5411 Machine Learning Project requirements, including data preprocessing, sequence generation, model development, evaluation, visualisation, and documentation.

## Project Overview

Grass minimum temperature is an important meteorological indicator reflecting radiative cooling and seasonal variation. The dataset is smooth, highly seasonal, and predictable, making it suitable for sequential deep learning models.

This project includes:

Training models using data from 1980–2024

Testing on unseen data from 2025-01-01 to 2025-10-30

Comparing RNN, LSTM, and BiLSTM performance

Analysing prediction errors

Documenting the entire workflow in GitHub

## Project Structure

Project/

daily_HKO_GMT_ALL.csv — Raw HKO dataset

processed_HKO_GMT_ALL.csv — Cleaned dataset

data_preprocessing.py — Cleaning, interpolation, datetime handling

sequence_generator.py — Sliding-window (45 days) sequence generation

models_rnn_lstm.py — RNN, LSTM, BiLSTM model definitions

train_and_evaluate.py — Training, evaluation, visualisation

hyperparameter_tuning.py — Optional: window sizes (30/45/60)

models/

rnn_model.h5

lstm_model.h5

bilstm_model.h5

figures/

actual_vs_predicted.png

error_distribution.png

## How to Run the Project

### 1. Install required libraries

Run the following command in your environment:
pip install pandas numpy scikit-learn tensorflow matplotlib

### 2. Preprocess the dataset

Run:
python data_preprocessing.py

This script loads the raw dataset, handles missing values with time-based interpolation, converts date fields to datetime format, and outputs the cleaned file processed_HKO_GMT_ALL.csv.

### 3. Train and evaluate the models

Run:
python train_and_evaluate.py

This script will:

Generate train/test sequences (45-day window → next-day prediction)

Train RNN, LSTM, and BiLSTM models

Evaluate each model using MAE and RMSE

Save trained models into the models/ directory

Produce visualisation plots in the figures/ directory

### 4. Optional hyperparameter tuning

Run:
python hyperparameter_tuning.py

This experiment tests different sliding-window lengths (30, 45, 60 days). The final project uses a 45-day window, which produced stable and consistent results.

## Evaluation Results (2025 Test Set)

Model	MAE (°C)	RMSE (°C)
RNN	1.02	1.35
LSTM	1.10	1.40
BiLSTM	1.14	1.42

Training configuration:

Optimizer: Adam

Loss: Mean Squared Error (MSE)

Epochs: 80

Batch size: 32

Sliding window: 45 days

## Visualisations

### actual_vs_predicted.png

A comparison of actual 2025 temperatures with predictions from RNN, LSTM, and BiLSTM. All models capture the overall seasonal trend, including winter lows and summer highs.

### error_distribution.png

A histogram showing prediction errors from all three models. This visualisation highlights days with unusually large deviations, often related to rapid temperature drops or seasonal transitions.

## Discussion

### Model Performance

The Simple RNN achieved the lowest MAE and RMSE. Given that grass minimum temperature is highly seasonal, smooth, and predictable, a simple recurrent structure generalises effectively without requiring complex memory mechanisms.

### Why LSTM and BiLSTM did not outperform RNN

This dataset exhibits:

Strong seasonal cycles

Limited irregularity

Low variance

Thus, the added complexity of LSTM and BiLSTM does not translate into better performance.

### Error Analysis

Larger prediction errors appear during:

Sudden temperature drops

Seasonal transitions

Isolated cold surge events

These are relatively rare in the training dataset, resulting in lower predictive accuracy during such events.

### Future Work

Possible improvements include:

Adding meteorological features (humidity, rainfall, wind speed)

Testing deeper LSTM or GRU networks

Exploring CNN–LSTM hybrid models

Implementing attention mechanisms

Further hyperparameter tuning

## Dataset Reference

Hong Kong Observatory — Daily Grass Minimum Temperature
https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp
