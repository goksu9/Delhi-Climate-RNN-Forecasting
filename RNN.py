import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv('DailyDelhiClimateTrain.csv')

# 1. Tarihi datetime'a çevir
df['date'] = pd.to_datetime(df['date'])

# 2. İndeksi tarih yap
df.set_index('date', inplace=True)       # inplace=True ⇒ df doğrudan güncellenir

# 3. Sadece ileri modellemede kullanacağımız sütunu bırak
df = df[['meantemp']]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)   # df yalnızca 'meantemp' içeriyor → ndarray


WINDOW = 5

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])   # geçmiş 5 gün
        y.append(data[i + window])       # ertesi gün
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, WINDOW)  # scaled = MinMax çıktısı
X = X.reshape((X.shape[0], X.shape[1], 1))   # (örnek, zaman, özellik)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

model = Sequential([
    SimpleRNN(50, activation="relu", input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs=20,            # daha uzun denemek istersen 50-100’e çıkar
    validation_data=(X_test, y_test),
    verbose=1
)


y_pred = model.predict(X_test)

y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"RMSE  : {rmse:.3f} °C")
print(f"MAE   : {mae:.3f} °C")

results = pd.DataFrame({
    "Gerçek": y_test_inv.flatten(),
    "Tahmin": y_pred_inv.flatten()
})
print(results.head(20))  # ilk 20 satır

plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label="Gerçek Değerler")
plt.plot(y_pred_inv, label="Tahminler")
plt.legend()
plt.xlabel("Zaman")
plt.ylabel("Sıcaklık (°C)")
plt.title("RNN Tahmin Sonuçları")
plt.show()


