import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import tensorflow as tf

# === CONFIG ===
CSV_PATH = "DailyDelhiClimateTrain.csv"  # gerekirse değiştir
WINDOW = 5
EPOCHS = 20  # dilersen yükselt (50-100 gibi)

# 1) Veriyi oku
df = pd.read_csv(CSV_PATH)

# 2) Tarihi datetime'a çevir
df['date'] = pd.to_datetime(df['date'])

# 3) İndeksi tarih yap
df.set_index('date', inplace=True)

# 4) Sadece kullanılacak sütunu bırak
df = df[['meantemp']]

# 5) Ölçekleme
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)  # ndarray

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, WINDOW)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (örnek, zaman, özellik)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 6) Model
model = Sequential([
    SimpleRNN(50, activation="relu", input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print(model.summary())

# 7) Eğitim
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=1
)

# 8) Değerlendirme
y_pred = model.predict(X_test)
# Ters ölçekleme için y şekillerini 2D'ye getir
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"RMSE  : {rmse:.3f} °C")
print(f"MAE   : {mae:.3f} °C")

# 9) İlk 20 sonuç
results = pd.DataFrame({
    "Gerçek": y_test_inv.flatten(),
    "Tahmin": y_pred_inv.flatten()
})
print(results.head(20))

# 10) Grafik
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="Gerçek Değerler")
plt.plot(y_pred_inv, label="Tahminler")
plt.legend()
plt.xlabel("Zaman")
plt.ylabel("Sıcaklık (°C)")
plt.title("RNN Tahmin Sonuçları")
plt.tight_layout()
plt.show()

# 11) MODELLERİ ve SCALER'I KAYDET
# Keras .h5
model.save("rnn_model.h5")
print("[+] Keras .h5 model kaydedildi: rnn_model.h5")

# Scaler
joblib.dump(scaler, "scaler.pkl")
print("[+] Scaler kaydedildi: scaler.pkl")

# 12) TFLITE ÇIKTILARI
# a) Ham (float32) TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("rnn_model.tflite", "wb") as f:
    f.write(tflite_model)
print("[+] TFLite model kaydedildi: rnn_model.tflite")

# b) Dinamik aralık kuantizasyonlu TFLite (daha küçük boyut, genelde benzer doğruluk)
converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_q_model = converter_q.convert()
with open("rnn_model_dynamic.tflite", "wb") as f:
    f.write(tflite_q_model)
print("[+] TFLite (dynamic range quantized) model kaydedildi: rnn_model_dynamic.tflite")

# 13) ÖRNEK INFERENCE FONKSİYONU (API'siz kullanım için)
def predict_next(meantemps_last5):
    """
    meantemps_last5: [t-4, t-3, t-2, t-1, t] (son 5 günün gerçek °C değeri)
    Keras .h5 model ile tahmin döndürür.
    """
    arr = np.array(meantemps_last5, dtype=np.float32).reshape(-1, 1)
    arr_scaled = scaler.transform(arr)  # (5,1)
    x = arr_scaled.reshape(1, WINDOW, 1)
    y_scaled = model.predict(x, verbose=0)  # (1,1)
    y = scaler.inverse_transform(y_scaled)[0, 0]
    return float(y)

if __name__ == "__main__":
    example = [28.1, 27.5, 29.0, 30.2, 31.1]
    print("Örnek tahmin (°C):", predict_next(example))
