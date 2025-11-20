# DIT5411 Machine Learning Project  
## Forecasting Hong Kong Daily Grass Minimum Temperature using RNN and LSTM

This project develops and compares RNN and LSTM models to forecast Hong Kong’s daily grass minimum temperature using historical data from the Hong Kong Observatory (HKO). The work follows the DIT5411 Machine Learning project requirements, including data preprocessing, model development, training, evaluation, visualization, and model comparison.

---

## Project Structure

Project/
│
├─ daily_HKO_GMT_ALL.csv              # Raw HKO dataset
├─ processed_HKO_GMT_ALL.csv          # Cleaned dataset
│
├─ data_preprocessing.py              # Data cleaning and date handling
├─ sequence_generator.py              # 45-day sliding window sequence generator
├─ models_rnn_lstm.py                 # RNN and LSTM model definitions
├─ train_and_evaluate.py              # Model training, evaluation, and plot generation
│
├─ models/
│   ├─ rnn_model.h5                   # Saved RNN model
│   └─ lstm_model.h5                  # Saved LSTM model
│
└─ figures/
    ├─ actual_vs_predicted.png
    └─ error_distribution.png

---

## How to Run

### 1. Install required libraries
pip install pandas numpy scikit-learn tensorflow matplotlib

### 2. Preprocess the raw dataset
python data_preprocessing.py
This will generate the file processed_HKO_GMT_ALL.csv.

### 3. Train the RNN and LSTM models
python train_and_evaluate.py

This script will:
- Train both models
- Evaluate them using MAE and RMSE
- Save the trained models in the models/ folder
- Generate two plots in the figures/ folder

---

## Results

### Evaluation Metrics
| Model | MAE (°C) | RMSE (°C) |
|-------|----------|-----------|
| RNN   | 0.94     | 1.30      |
| LSTM  | 1.63     | 1.85      |

---

## Generated Plots

actual_vs_predicted.png  
Displays the actual 2025 daily grass minimum temperature and the predictions from both RNN and LSTM models.

error_distribution.png  
Shows the distribution of prediction errors (RNN vs LSTM).

---

## Discussion

- The RNN model performed better than the LSTM model in this project.
- The dataset has strong seasonality and smooth temperature transitions, which simple RNN architectures can learn effectively.
- The LSTM model may require deeper layers, more training, or hyperparameter tuning to outperform the RNN.
- Both models capture the general seasonal trend, but LSTM predictions show larger deviations.

---

## Reference

Hong Kong Observatory — Daily Grass Minimum Temperature Dataset  
https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp
