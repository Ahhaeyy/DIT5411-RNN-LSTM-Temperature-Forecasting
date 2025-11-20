# DIT5411 Machine Learning Project  
## Forecasting Hong Kong Daily Grass Minimum Temperature using RNN, LSTM and BiLSTM

This project develops and compares three deep learning models to forecast Hong Kong’s daily grass minimum temperature using historical data from the Hong Kong Observatory (HKO). The project follows the requirements of DIT5411 Machine Learning and includes data preprocessing, sequence generation, model development, training, evaluation, visualization, and hyperparameter tuning.

---

## Project Structure

Project/
│
├─ daily_HKO_GMT_ALL.csv              # Raw HKO dataset
├─ processed_HKO_GMT_ALL.csv          # Cleaned dataset
│
├─ data_preprocessing.py              # Data cleaning and date handling
├─ sequence_generator.py              # 45-day sliding window sequence generator
├─ models_rnn_lstm.py                 # RNN, LSTM and Bidirectional LSTM model definitions
├─ train_and_evaluate.py              # Training, evaluation and plot generation
├─ hyperparameter_tuning.py (optional) # Simple experiment for different window lengths
│
├─ models/
│   ├─ rnn_model.h5                   # Saved RNN model
│   ├─ lstm_model.h5                  # Saved LSTM model
│   └─ bilstm_model.h5                # Saved Bidirectional LSTM model
│
└─ figures/
    ├─ actual_vs_predicted.png
    └─ error_distribution.png

---

## How to Run

1. Install required libraries:
pip install pandas numpy scikit-learn tensorflow matplotlib

2. Preprocess the raw dataset:
python data_preprocessing.py

3. Train and evaluate the models:
python train_and_evaluate.py
This script will train three models (RNN, LSTM, BiLSTM), evaluate using MAE and RMSE, save trained models in the models/ folder, and generate plots in the figures/ folder.

4. (Optional) Hyperparameter tuning:
python hyperparameter_tuning.py
This simple experiment tests different sequence lengths (30, 45, 60 days) and reports MAE. The chosen window size (45) was selected based on this result.

---

## Results

### Evaluation Metrics

| Model    | MAE (°C) | RMSE (°C) |
|----------|----------|-----------|
| RNN      | 1.02     | 1.35      |
| LSTM     | 1.10     | 1.40      |
| BiLSTM   | 1.14     | 1.42      |

### Visualizations
- actual_vs_predicted.png: Actual vs predicted daily grass minimum temperature for 2025
- error_distribution.png: Histogram of prediction errors for each model

---

## Discussion

- The BiLSTM model performed slightly worse than the simpler RNN for this dataset.
- The dataset shows strong seasonality and fairly stable variation, so a simpler model (RNN) learned the pattern well.
- More complex models like LSTM and BiLSTM may need more data or tuning to outperform simpler ones.
- Future work could include deeper architectures, additional weather features (humidity, wind), or other sequence lengths.

---

## Reference

Hong Kong Observatory. (2025). Daily grass minimum temperature dataset.  
https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp
