import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ——— Load actual & predicted arrays ———
# From your NumPy LSTM script
numpy_preds = np.load('numpy_lstm_predictions.npy')
y_test_numpy = np.load('y_test.npy')

# From your Keras script (ensure you saved these before exiting)
keras_preds = {
    'LSTM': lstm_predictions,
    'RNN': rnn_predictions,
    'Bidirectional RNN': bidirectional_rnn_predictions,
    'Dense': dense_predictions,
}
y_test_keras = y_test  # already inverse‑scaled

# ——— Build a unified DataFrame indexed by timestamp ———
timestamps = df.index[train_size + seq_length:]  # same index for both
df_plot = pd.DataFrame({
    'Actual': y_test_numpy.flatten(),
    'NumPy LSTM': numpy_preds.flatten(),
})
for name, preds in keras_preds.items():
    df_plot[name] = preds.flatten()

df_plot.index = timestamps

# ——— Plot everything in one figure ———
plt.figure(figsize=(16, 8))
for col in df_plot.columns:
    plt.plot(df_plot.index, df_plot[col], label=col)
plt.title('Actual vs Predicted — All Models')
plt.xlabel('Timestamp')
plt.ylabel('Discharge Value')
plt.legend()
plt.grid(True)
plt.show()
