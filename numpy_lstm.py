import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ingestor import Ingestor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Set random seed for reproducibility
np.random.seed(40) #30

# Disable TensorFlow OneDNN optimizations (if needed for compatibility)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load data using Ingestor
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'
data = Ingestor(amnistadRelease).data

df = pd.DataFrame(data)

# Ensure Timestamp is in datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Set Timestamp as index
df.set_index('Timestamp', inplace=True)

# Handle missing values in 'Value' column
df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(method='ffill')

# Extract time-based features
df['CustomTimeFeature'] = (
        df.index.year * 10000 +
        df.index.month * 100 +
        df.index.day +
        df.index.hour / 24 +
        df.index.minute / 1440
)

# Scale the features using MinMaxScaler
scaler_time = MinMaxScaler()
scaler_values = MinMaxScaler()

df['ScaledTime'] = scaler_time.fit_transform(df[['CustomTimeFeature']])
df['ScaledValue'] = scaler_values.fit_transform(df[['Value']])

combined_features = df[['ScaledTime', 'ScaledValue']].values


# Function to create sequences for training
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(seq_length, len(data)):
        sequences.append(data[i - seq_length:i])
        targets.append(data[i, 1])  # Predicting 'ScaledValue'
    return np.array(sequences), np.array(targets).reshape(-1, 1)


seq_length = 5
X, y = create_sequences(combined_features, seq_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]


# --------------------------- Improved NumPy LSTM Model --------------------------- #
class NumPyLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Xavier Initialization for weights
        def xavier_init(shape):
            return np.random.randn(*shape) * np.sqrt(1.0 / shape[1])

        # Initialize weights and biases for gates
        self.W = {gate: xavier_init((hidden_dim, hidden_dim + input_dim + 1)) for gate in ['f', 'i', 'c', 'o']}
        self.b = {gate: np.zeros((hidden_dim, 1)) for gate in ['f', 'i', 'c', 'o']}

        # Output layer weights
        self.W['y'] = xavier_init((output_dim, hidden_dim))
        self.b['y'] = np.zeros((output_dim, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Prevent overflow

    @staticmethod
    def tanh(x):
        return np.tanh(np.clip(x, -500, 500))  # Prevent overflow

    def forward(self, x, h_t=None, c_t=None, return_combined=False):
        T, _ = x.shape
        if h_t is None:
            h_t = np.zeros((self.hidden_dim, 1))
            c_t = np.zeros((self.hidden_dim, 1))

        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            combined = np.vstack((h_t, np.ones((1, 1)), x_t))

            f_t = self.sigmoid(np.dot(self.W['f'], combined) + self.b['f'])
            i_t = self.sigmoid(np.dot(self.W['i'], combined) + self.b['i'])
            o_t = self.sigmoid(np.dot(self.W['o'], combined) + self.b['o'])
            c_t_candidate = self.tanh(np.dot(self.W['c'], combined) + self.b['c'])

            c_t = f_t * c_t + i_t * c_t_candidate
            h_t = o_t * self.tanh(c_t)

        y_t = np.dot(self.W['y'], h_t) + self.b['y']

        if return_combined:
            return y_t.flatten(), h_t, c_t, combined
        return y_t.flatten(), h_t, c_t

    def train(self, X_train, y_train, epochs=50):
        print("\nTraining NumPy LSTM model with Backpropagation...")

        for epoch in range(epochs):
            total_loss = 0
            h_t = np.zeros((self.hidden_dim, 1))
            c_t = np.zeros((self.hidden_dim, 1))

            for i in range(len(X_train)):
                x, y = X_train[i], y_train[i].reshape(-1, 1)
                y_pred, h_t, c_t, combined_t = self.forward(x, h_t, c_t, return_combined=True)
                loss = np.mean((y_pred - y) ** 2)  # MSE Loss
                total_loss += loss

                # Compute gradient
                error = y_pred - y
                gradient_y = error

                # Update output layer weights
                self.W['y'] -= self.learning_rate * np.dot(gradient_y, h_t.T)
                self.b['y'] -= self.learning_rate * gradient_y

                # Backpropagation for LSTM gates (Fix: Use combined_t for shape alignment)
                dh = np.dot(self.W['y'].T, gradient_y)

                for gate in ['f', 'i', 'c', 'o']:
                    dW_gate = np.dot(dh, combined_t.T)  # Fix shape alignment
                    self.W[gate] -= self.learning_rate * dW_gate
                    self.b[gate] -= self.learning_rate * dh.mean(axis=1, keepdims=True)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.6f}")


# Initialize and train the improved NumPy LSTM model
lstm_model = NumPyLSTM(input_dim=2, hidden_dim=50, output_dim=1, learning_rate=0.001)
lstm_model.train(X_train, y_train, epochs=50)

# Make predictions with the NumPy LSTM model
# Make predictions with the NumPy LSTM model
lstm_predictions = [lstm_model.forward(x_test)[0] for x_test in X_test]  # Extract only y_t
lstm_predictions = np.array(lstm_predictions).reshape(-1, 1)
lstm_predictions = scaler_values.inverse_transform(lstm_predictions)




# Inverse transform y_test
y_test = scaler_values.inverse_transform(y_test)
np.save('numpy_lstm_predictions.npy', lstm_predictions)
np.save('y_test.npy', y_test)

# ---------------------- Evaluation Metrics ---------------------- #
lstm_mse = mean_squared_error(y_test, lstm_predictions)
lstm_mae = mean_absolute_error(y_test, lstm_predictions)

print(f'LSTM Mean Squared Error: {lstm_mse:.6f}')
print(f'LSTM Mean Absolute Error: {lstm_mae:.6f}')

# ---------------------- Plot Results ---------------------- #
plt.figure(figsize=(14, 7))
plt.plot(df.index[train_size + seq_length:], y_test, label='Actual', color='black')
plt.plot(df.index[train_size + seq_length:], lstm_predictions, label='NumPy LSTM Predicted', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Prediction vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()

