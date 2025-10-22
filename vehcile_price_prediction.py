import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'Year': [2010, 2012, 2015, 2018, 2020, 2021, 2014, 2019, 2017, 2022],
    'Kilometers_Driven': [120000, 90000, 40000, 20000, 15000, 10000, 60000, 25000, 30000, 8000],
    'Fuel_Type': [0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    'Selling_Price': [3.5, 4.0, 6.0, 8.5, 9.5, 10.0, 5.0, 9.0, 7.5, 11.0]
}

df = pd.DataFrame(data)

X = df[["Year", "Kilometers_Driven", "Fuel_Type"]].values
y = df["Selling_Price"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, y_train, epochs=100, verbose=0)

plt.plot(history.history["loss"])
plt.title("Training Loss Over Time")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.2f}")

new_car = np.array([[2024, 5500, 0]])
new_car_scaled = scaler.transform(new_car)
predicted_price = model.predict(new_car_scaled)
print(f"Predicted Selling Price: {predicted_price[0][0]:.2f} lakhs")
