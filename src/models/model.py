import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# LSTM model
def build_model(input_shape=(10,4)):
    model=Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(1) # Output layer (regression)
    ])
    # Compile the model
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
    
    return model
    
