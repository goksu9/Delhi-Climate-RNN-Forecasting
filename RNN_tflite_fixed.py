import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import tensorflow as tf

# === CONFIG ===
CSV_PATH = "DailyDelhiClimateTrain.csv"
WINDOW = 5
EPOCHS = 20

# 1) Load data
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df[['meantemp']]

# 2) Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, WINDOW)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 3) Model (use unroll=True to avoid TensorArray/TensorList in graph)
model = Sequential([
    SimpleRNN(50, activation="relu", unroll=True, input_shape=(WINDOW, 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print(model.summary())

# 4) Train
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=1
)

# 5) Evaluate
y_pred = model.predict(X_test, verbose=0)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"RMSE  : {rmse:.3f} °C")
print(f"MAE   : {mae:.3f} °C")

# 6) Save artifacts
model.save("rnn_model.keras")  # modern Keras format
print("[+] Keras model saved: rnn_model.keras")
joblib.dump(scaler, "scaler.pkl")
print("[+] Scaler saved: scaler.pkl")

# 7) TFLite conversion with graceful fallback
def convert_tflite(m, filename, quantize=False):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(filename, "wb") as f:
            f.write(tflite_model)
        print(f"[+] TFLite saved (BUILTINS): {filename}")
        return True
    except Exception as e:
        print(f"[!] BUILTINS conversion failed: {e}")
        print("[i] Retrying with SELECT_TF_OPS and disabling tensor list lowering...")
        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # Disable lowering TensorList ops to avoid TensorListReserve error
        try:
            converter._experimental_lower_tensor_list_ops = False  # private flag, but practical
        except Exception:
            pass
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(filename, "wb") as f:
            f.write(tflite_model)
        print(f"[+] TFLite saved (SELECT_TF_OPS): {filename}")
        print("[!] Note: SELECT_TF_OPS requires the TF Lite Flex delegate at runtime (e.g., include 'tensorflow-lite-select-tf-ops').")
        return True

# Plain float32
convert_tflite(model, "rnn_model.tflite", quantize=False)
# Dynamic range quantized
convert_tflite(model, "rnn_model_dynamic.tflite", quantize=True)

# 8) Simple local inference helper
def predict_next(meantemps_last5):
    arr = np.array(meantemps_last5, dtype=np.float32).reshape(-1, 1)
    arr_scaled = scaler.transform(arr)
    x = arr_scaled.reshape(1, WINDOW, 1)
    y_scaled = model.predict(x, verbose=0)
    y = scaler.inverse_transform(y_scaled)[0, 0]
    return float(y)

if __name__ == "__main__":
    example = [28.1, 27.5, 29.0, 30.2, 31.1]
    print("Example prediction (°C):", predict_next(example))
