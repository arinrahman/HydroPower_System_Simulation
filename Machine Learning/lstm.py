import tensorflow as tf
import matplotlib.pyplot as plt
from Ingestor import Ingestor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

def create_sequences_shift_left(data, seq_length, shift=1):
    """
    Creates sequences and shifts the target (y) by `shift` steps to the left.
    The `shift` value determines how far back the target value will be.
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - shift):
        sequence = data[i:i + seq_length]
        target = data[i + seq_length - shift]  # Shift the target
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


# Define sequence length and shift
seq_length = 3
shift = 1  # This will shift the target by one step to the left

# Create sequences with shifted target
X, y = create_sequences_shift_left(scaled_data, seq_length, shift)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape for LSTM and RNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define and train LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

lstm_model.fit(X_train, y_train, epochs=75, batch_size=3, validation_data=(X_test, y_test))

# Define and train RNN model
rnn_model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    SimpleRNN(50),
    Dropout(0.2),
    Dense(1)
])

rnn_optimizer = Adam(learning_rate=0.001)
rnn_model.compile(optimizer=rnn_optimizer, loss='mean_squared_error')

rnn_model.fit(X_train, y_train, epochs=75, batch_size=3, validation_data=(X_test, y_test))

# Define and train Basic Neural Network model
# Flatten X_train and X_test for Dense model
X_train_dense = X_train.reshape((X_train.shape[0], -1))  # Flatten the sequence
X_test_dense = X_test.reshape((X_test.shape[0], -1))

dense_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_dense.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

dense_optimizer = Adam(learning_rate=0.001)
dense_model.compile(optimizer=dense_optimizer, loss='mean_squared_error')

dense_model.fit(X_train_dense, y_train, epochs=75, batch_size=3, validation_data=(X_test_dense, y_test))

# Make predictions for the test set with LSTM model
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = lstm_predictions.reshape(-1, 1)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Make predictions for the test set with RNN model
rnn_predictions = rnn_model.predict(X_test)
rnn_predictions = rnn_predictions.reshape(-1, 1)
rnn_predictions = scaler.inverse_transform(rnn_predictions)

# Make predictions for the test set with Dense model
dense_predictions = dense_model.predict(X_test_dense)
dense_predictions = dense_predictions.reshape(-1, 1)
dense_predictions = scaler.inverse_transform(dense_predictions)

# Inverse transform y_test back to original values
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and print metrics for LSTM model
lstm_mse = mean_squared_error(y_test, lstm_predictions)
print(f'LSTM Mean Squared Error: {lstm_mse}')

lstm_mae = mean_absolute_error(y_test, lstm_predictions)
print(f'LSTM Mean Absolute Error: {lstm_mae}')

# Calculate and print metrics for RNN model
rnn_mse = mean_squared_error(y_test, rnn_predictions)
print(f'RNN Mean Squared Error: {rnn_mse}')

rnn_mae = mean_absolute_error(y_test, rnn_predictions)
print(f'RNN Mean Absolute Error: {rnn_mae}')

# Calculate and print metrics for Dense model
dense_mse = mean_squared_error(y_test, dense_predictions)
print(f'Dense Mean Squared Error: {dense_mse}')

dense_mae = mean_absolute_error(y_test, dense_predictions)
print(f'Dense Mean Absolute Error: {dense_mae}')

# Calculate average gap for all models
lstm_absolute_errors = np.abs(y_test - lstm_predictions)
lstm_average_gap = np.mean(lstm_absolute_errors)

rnn_absolute_errors = np.abs(y_test - rnn_predictions)
rnn_average_gap = np.mean(rnn_absolute_errors)

dense_absolute_errors = np.abs(y_test - dense_predictions)
dense_average_gap = np.mean(dense_absolute_errors)

print(f'LSTM Average Gap: {lstm_average_gap}')
print(f'RNN Average Gap: {rnn_average_gap}')
print(f'Dense Average Gap: {dense_average_gap}')

# Custom prediction
custom_timestamp = '2024-06-21 19:30:00'
custom_value = 406.8
predicted_custom_value_lstm = predict_with_custom_value(custom_timestamp, custom_value, scaler, lstm_model, seq_length)
predicted_custom_value_rnn = predict_with_custom_value(custom_timestamp, custom_value, scaler, rnn_model, seq_length)

# Prepare input for Dense model
# Use the last sequence and append the scaled custom value
scaled_custom_value = scaler.transform(np.array([[custom_value]]))

# Ensure the input is reshaped to match the Dense model's input shape
custom_sequence_dense = np.append(scaled_data[-(seq_length - 1):], scaled_custom_value).reshape(1, -1)

# Ensure correct input shape
if custom_sequence_dense.shape[1] != X_train_dense.shape[1]:
    raise ValueError(f"Expected input shape: {X_train_dense.shape[1]}, but got: {custom_sequence_dense.shape[1]}")

predicted_custom_value_dense = dense_model.predict(custom_sequence_dense)
predicted_custom_value_dense = scaler.inverse_transform(predicted_custom_value_dense)

custom_timestamp = pd.to_datetime(custom_timestamp)

# Plot results
test_timestamps = df.index[train_size + seq_length + shift - 1 : train_size + seq_length + shift - 1 + len(y_test)]
y_test = y_test.reshape(-1)

plt.figure(figsize=(14, 7))

plt.plot(test_timestamps, y_test, label='Actual', color='black')
plt.plot(test_timestamps, lstm_predictions, label='LSTM Predicted', color='blue')
plt.plot(test_timestamps, rnn_predictions, label='RNN Predicted', color='orange')
plt.plot(test_timestamps, dense_predictions, label='Dense Predicted', color='green')

plt.scatter([custom_timestamp], [predicted_custom_value_lstm], color='red', label='LSTM Custom Prediction', zorder=5)
plt.scatter([custom_timestamp], [predicted_custom_value_rnn], color='purple', label='RNN Custom Prediction', zorder=5)
plt.scatter([custom_timestamp], [predicted_custom_value_dense], color='blue', label='Dense Custom Prediction', zorder=5)

plt.text(
    0.02, 0.95, 
    f'LSTM Average Gap: {lstm_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.text(
    0.02, 0.90, 
    f'RNN Average Gap: {rnn_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.text(
    0.02, 0.85, 
    f'Dense Average Gap: {dense_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Prediction vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()

# Ensure residuals (errors) for each model are calculated as 1D arrays
lstm_errors = y_test - lstm_predictions.reshape(-1)
rnn_errors = y_test - rnn_predictions.reshape(-1)
dense_errors = y_test - dense_predictions.reshape(-1)

# Check that test_timestamps matches the number of residuals
if len(test_timestamps) != len(lstm_errors):
    raise ValueError(f"Mismatch between number of timestamps ({len(test_timestamps)}) and errors ({len(lstm_errors)})")

# Plot the errors (residuals) over time
plt.figure(figsize=(14, 7))

plt.plot(test_timestamps, lstm_errors, label='LSTM Error', color='blue')
plt.plot(test_timestamps, rnn_errors, label='RNN Error', color='orange')
plt.plot(test_timestamps, dense_errors, label='Dense Error', color='green')

plt.axhline(0, color='black', linestyle='--')  # Add a horizontal line at y=0 for reference

plt.xlabel('Timestamp')
plt.ylabel('Error (Actual - Predicted)')
plt.title('Prediction Errors Over Time')
plt.legend()
plt.grid(True)
plt.show()