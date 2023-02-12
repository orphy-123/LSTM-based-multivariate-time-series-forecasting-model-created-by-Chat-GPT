import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Pre-defined dataset
data = np.array([[i+j for j in range(5)] for i in range(100)])

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into train, validation, and test sets
train_data, test_data = train_test_split(data_scaled, shuffle=False, test_size=0.1)
val_data, test_data = train_test_split(test_data, shuffle=False, test_size=0.5)

# Create a function to split the data into windows of size `window_size`
def window_data(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :-1])
        y.append(data[i+window_size, -1])
    return np.array(X), np.array(y)

# Define the window size
window_size = 5

# Split the train, validation, and test sets into windows
X_train, y_train = window_data(train_data, window_size)
X_val, y_val = window_data(val_data, window_size)
X_test, y_test = window_data(test_data, window_size)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32)

# Predict on the test set and evaluate using RMSE
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean(np.square(y_pred - y_test)))
print("Test RMSE:", rmse)