<h1>Hong Kong Daily Minimum Grass Temperature Forecasting</h1>

<h2>Project Overview</h2>
This project applies Deep Learning techniques to forecast daily grass minimum temperatures in Hong Kong. As part of the <strong>DIT5411 Machine Learning</strong> course, we implement and compare three different time-series models:
1.  <strong>Simple RNN</strong> (Baseline)
2.  <strong>LSTM</strong> (Long Short-Term Memory)
3.  <strong>BiLSTM</strong> (Bidirectional LSTM)

The goal is to analyze historical data (1980-2024) to predict temperature trends for the "future" period (Jan - Oct 2025).

<h2>Dataset</h2>
<ul>
  <li><strong>Source</strong>: Hong Kong Observatory via <a href="https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-grass-min-temp">data.gov.hk</a></li>
  <li><strong>Target Variable</strong>: Daily Grass Minimum Temperature (°C)</li>
  <li><strong>Data Split</strong>:
    <ul>
      <li><strong>Training</strong>: 1980 - 2024</li>
      <li><strong>Testing</strong>: Jan 1, 2025 - Oct 30, 2025</li>
    </ul>
  </li>
</ul>

<h2>Repository Structure</h2>
<pre>
├── daily_HKO_GMT_ALL.csv      # Raw dataset (Input)
├── processed_HKO_GMT_ALL.csv  # Preprocessed dataset (Output of preprocessing)
├── data_preprocessing.py      # Script for data cleaning, interpolation, and normalization
├── sequence_generator.py      # Helper for creating sliding window sequences
├── models_rnn_lstm.py         # Keras definitions for RNN, LSTM, and BiLSTM architectures
├── train_and_evaluate.py      # Main script: trains models and generates evaluation metrics/plots
├── hyperparameter_tuning.py   # Script for finding optimal model parameters
├── rnn_model.h5               # Saved weights for the RNN model
├── lstm_model.h5              # Saved weights for the LSTM model
├── bilstm_model.h5            # Saved weights for the BiLSTM model
└── README.md                  # Project documentation
</pre>

<h2>Installation</h2>
Ensure you have Python 3.8+ installed. Install the required dependencies:

<pre>
pip install numpy pandas matplotlib scikit-learn tensorflow
</pre>

<h2>Usage Instructions</h2>

<h3>Step 1: Data Preprocessing</h3>
Run the preprocessing script to clean the raw data, handle missing values, and normalize features.
<pre>
python data_preprocessing.py
</pre>
<em>Output: <code>processed_HKO_GMT_ALL.csv</code></em>

<h3>Step 2: Training and Evaluation</h3>
Run the main script to train all three models (RNN, LSTM, BiLSTM) and evaluate them on the 2025 test set. This will also generate comparison plots.
<pre>
python train_and_evaluate.py
</pre>

<h3>Step 3: Hyperparameter Tuning (Optional)</h3>
If you wish to experiment with different configurations:
<pre>
python hyperparameter_tuning.py
</pre>

<h2>Results (Test Period: 2025)</h2>

Models were evaluated using <strong>Mean Absolute Error (MAE)</strong> and <strong>Root Mean Squared Error (RMSE)</strong>.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>MAE (°C)</th>
      <th>RMSE (°C)</th>
      <th>Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>RNN (Baseline)</strong></td>
      <td>1.4411</td>
      <td>1.7119</td>
      <td>Baseline</td>
    </tr>
    <tr>
      <td><strong>LSTM</strong></td>
      <td><strong>0.9216</strong></td>
      <td><strong>1.2611</strong></td>
      <td><strong>Best</strong></td>
    </tr>
    <tr>
      <td><strong>BiLSTM</strong></td>
      <td>0.9520</td>
      <td>1.2650</td>
      <td>Competitive</td>
    </tr>
  </tbody>
</table>

<h3>Analysis</h3>
<ul>
  <li><strong>LSTM</strong> achieved the best performance, significantly outperforming the simple RNN. This demonstrates the LSTM's superior ability to capture long-term dependencies and seasonal patterns in meteorological data over a 60-day lookback window.</li>
  <li><strong>BiLSTM</strong> performed similarly to the standard LSTM but did not yield a significant improvement, suggesting that future context within the training window was less critical for this specific forecasting task.</li>
</ul>

<h2>Contributors</h2>
<ul>
  <li><strong>[Your Name]</strong> - [Student ID]</li>
  <li><strong>[Teammate Name]</strong> - [Student ID]</li>
  <li><strong>[Teammate Name]</strong> - [Student ID]</li>
</ul>

<h2>License</h2>
This project is for educational purposes. Data provided by the Hong Kong Observatory under the Open Data License.