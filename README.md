 🌡️ Delhi Climate Temperature Forecasting using RNN

This repository contains an end-to-end deep learning workflow for temperature forecasting using the **Daily Delhi Climate Dataset**.
A **SimpleRNN** neural network is trained to predict the next day's temperature based on the previous 5 days.
The project includes training scripts, evaluation results, model saving, and **TensorFlow Lite** conversion.

## 📁 Project Structure
DailyDelhiClimateTrain.csv
DailyDelhiClimateTest.csv
RNN.py
RNN_tflite.py
RNN_tflite_fixed.py
rnn_model.h5
rnn_model.keras
rnn_model.tflite
rnn_model_dynamic.tflite
scaler.pkl

## 🧠 Model Details
- Model Type: SimpleRNN Neural Network
- Input Window: Last 5 days temperature
- Prediction: Next-day temperature
- Frameworks: TensorFlow / Keras
- Metrics: MSE, MAE, RMSE

Model architecture:
SimpleRNN(50, activation="relu")
Dense(1)

## 🚀 How to Run
Install dependencies:
pip install -r requirements.txt

Train & Evaluate the Model:
python RNN.py

Convert to TensorFlow Lite:
python RNN_tflite.py

## 📊 Example Output
RMSE : ~X.XXX °C
MAE  : ~X.XXX °C

## 📦 Artifacts
- rnn_model.h5 → Standard Keras model
- rnn_model.keras → New Keras format
- rnn_model.tflite → TFLite model
- rnn_model_dynamic.tflite → Quantized model
- scaler.pkl → PreprocessingScaler

## 🔧 Example Inference Function
predict_next([28.1, 27.5, 29.0, 30.2, 31.1])

## 🎯 Future Work
- Add LSTM / GRU comparison
- Deploy via FastAPI
- Mobile deployment sample

## ✨ Author
Hüseyin Göksu Hacıoğlu




