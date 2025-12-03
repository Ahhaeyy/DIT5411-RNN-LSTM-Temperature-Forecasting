# DIT5411 Machine Learning Project  
## Forecasting Hong Kong Daily Grass Minimum Temperature using RNN, LSTM and BiLSTM

This project develops and compares three deep learning models—RNN, LSTM, and Bidirectional LSTM—to forecast Hong Kong’s daily grass minimum temperature using historical data from the Hong Kong Observatory (HKO). The work follows the DIT5411 Machine Learning project requirements, including data preprocessing, sequence generation, model building, training, evaluation, visualization, and model comparison.

---

## Project Structure

Project/
│
├─ daily_HKO_GMT_ALL.csv              # Raw HKO dataset  
├─ processed_HKO_GMT_ALL.csv          # Cleaned dataset  
│
├─ data_preprocessing.py              # Data cleaning and date handling  
├─ sequence_generator.py              # 45-day sliding window sequence generator  
├─ models_rnn_lstm.py                 # RNN, LSTM and BiLSTM model definitions  
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

1. **Install required libraries**  
pip install pandas numpy scikit-learn tensorflow matplotlib

2. **Preprocess the raw dataset**  
python data_preprocessing.py  
This will generate the cleaned file `processed_HKO_GMT_ALL.csv`.

3. **Train the RNN, LSTM and BiLSTM models**  
python train_and_evaluate.py  

This script will:  
- Train all three models  
- Evaluate them using MAE and RMSE  
- Save each trained model in the `models/` folder  
- Generate two visualisations in the `figures/` folder  

4. **(Optional) Run hyperparameter tuning**  
python hyperparameter_tuning.py  
This tests sequence lengths (30, 45, 60). The final project uses a 45-day window.

---

## Results

### Evaluation Metrics

| Model    | MAE (°C) | RMSE (°C) |
|----------|----------|-----------|
| RNN      | 1.02     | 1.35      |
| LSTM     | 1.10     | 1.40      |
| BiLSTM   | 1.14     | 1.42      |

These values come from training with 80 epochs, Adam optimizer, and MSE loss.

---

## Generated Plots

**actual_vs_predicted.png**  
Shows actual 2025 grass minimum temperatures and predicted curves from RNN, LSTM, and BiLSTM.

**error_distribution.png**  
Histogram of prediction errors for all three models.

---

## Discussion

- The RNN model performed the best among the three models in this project.  
- All models successfully captured the overall seasonal temperature trend, which is smooth and predictable.  
- The simpler RNN architecture handled the dataset well because the temperature time series has steady and regular patterns.  
- LSTM and BiLSTM are more complex models. They usually perform better on irregular or highly nonlinear sequences. With this relatively smooth dataset, the added complexity did not produce significant improvement.  
- The LSTM and BiLSTM models may require more layers, more neurons, or different hyperparameters to outperform RNN.  
- Future improvements could include:  
  - adding other weather features (humidity, wind, rainfall),  
  - deeper LSTM stacks,  
  - trying GRU or 1D-CNN models,  
  - tuning learning rates, batch sizes, or window sizes.  

---

## Reference

Hong Kong Observatory — Daily Grass Minimum Temperature Dataset  
https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp
