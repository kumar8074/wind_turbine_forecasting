import os
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        return joblib.load(file_path)
    except Exception as e:
        raise e

def load_keras_model(load_path: str):
    """
    Loads a Keras model saved in `.keras` format.

    Args:
        load_path (str): Path to the saved `.keras` model file.

    Returns:
        tf.keras.Model: The loaded model.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No model found at: {load_path}")
    if not load_path.endswith(".keras"):
        raise ValueError("Expected a `.keras` model file.")
    
    model = load_model(load_path)
    return model


def prepare_sequence_from_user_input(user_input_dict, csv_path, scaler, features, time_step=10):
    """
    Constructs an input sequence using last (time_step - 1) rows from original data + 1 user row
    """
    df = pd.read_csv(csv_path)
    df = df[features]
    last_n_rows = df.tail(time_step - 1)

    user_row_df = pd.DataFrame([user_input_dict])[features]

    combined = pd.concat([last_n_rows, user_row_df], ignore_index=True)
    assert combined.shape[0] == time_step, "Insufficient rows to construct input sequence"

    # Scale
    combined_scaled = scaler.transform(combined)
    final_sequence= np.expand_dims(combined_scaled, axis=0)  # shape: (1, time_step, num_features)
    return final_sequence

