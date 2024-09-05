import tensorflow as tf
import matplotlib.pyplot as plt
from Ingestor import Ingestor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and preprocess data
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'
data = Ingestor(amnistadRelease).data
df = pd.DataFrame(data)

# Convert 'Timestamp' to datetime and set as index
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Value']])

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length + 1]
        sequences.append(sequence)
    return np.array(sequences)

def predict_with_custom_value(timestamp, value, scaler, model, seq_length):
    timestamp = pd.to_datetime(timestamp)
    
    scaled_value = scaler.transform(np.array([[value]]))

    last_sequence = scaled_data[-(seq_length - 1):]

    custom_sequence = np.append(last_sequence, scaled_value).reshape((1, seq_length, 1))

    prediction = model.predict(custom_sequence)

    actual_prediction = scaler.inverse_transform(prediction)
    
    print(f"Prediction for the next time step after {timestamp} with input value {value}: {actual_prediction[0][0]}")
    return actual_prediction[0][0]

# Define sequence length and create sequences
seq_length = 3
sequences = create_sequences(scaled_data, seq_length)

# Define the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Split data into training and testing sets
train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]
X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, epochs=75, batch_size=3, validation_data=(X_test, y_test))

# Make predictions for the test set
predictions = model.predict(X_test)
predictions = predictions.reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

# Inverse transform y_test back to original values
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and print metrics
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Custom prediction
custom_timestamp = '2024-06-20 12:00:00'
custom_value = 270.8
predicted_custom_value = predict_with_custom_value(custom_timestamp, custom_value, scaler, model, seq_length)

custom_timestamp = pd.to_datetime(custom_timestamp)

test_timestamps = df.index[train_size + seq_length:]

# Config plot
plt.figure(figsize=(12, 6))

plt.plot(test_timestamps, y_test, label='Actual')

plt.plot(test_timestamps, predictions, label='Predicted')

plt.scatter([custom_timestamp], [predicted_custom_value], color='red', label='Custom Prediction', zorder=5)

plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values Over Time (Including Custom Prediction)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
