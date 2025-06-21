import numpy as np

# Create sequences of X, y pairs (for train,val & test data)
def create_sequences(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])  # Assuming the first feature is the target
    return np.array(X), np.array(y)