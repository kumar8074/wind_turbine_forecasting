import os
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import r2_score

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

def calculate_r2_score(y_true, y_pred, num_features, processor):
    """
    Calculate the R2 score for the predictions.
                
    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        num_features (int): Number of features in the data.
        processor (object): Preprocessor object for inverse transformation.
                
    Returns:
        float: R2 score.
    """
    # Flatten the predictions
    y_pred=y_pred.flatten()
                
    # Create an empty full array for inverse transformation
    y_pred_full=np.zeros((y_pred.shape[0],num_features))
                
    # Insert predictions into the first column
    y_pred_full[:, 0]=y_pred
                
    # Apply inverse transformation
    y_pred_inverse = processor.inverse_transform(y_pred_full)[:, 0]  # Extract only the first column
                
    # If y_test was also scaled, inverse transform it for comparison
    y_true_full = np.zeros((y_true.shape[0], num_features))  # Use the same number of features
    y_true_full[:,0]=y_true
                
    y_true_inverse= processor.inverse_transform(y_true_full)[:, 0]  # Extract only the first column
                
    # Calculate R2 Score
    r2= r2_score(y_true_inverse, y_pred_inverse)
                
    return r2
