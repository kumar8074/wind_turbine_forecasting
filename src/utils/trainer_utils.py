import os
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def save_keras_model(model: Model, save_path: str) -> None:
    """
    Saves the given model in the native `.keras` format.

    Args:
        model (tf.keras.Model): The trained model to save.
        save_path (str): Path to save the model, including `.keras` extension.
    """
    if not save_path.endswith(".keras"):
        raise ValueError("The save path must end with `.keras` extension.")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
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
        y_true_inverse (np.ndarray): Inverse transformed true values.
        y_pred_inverse (np.ndarray): Inverse transformed predicted values.
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
                
    return r2, y_true_inverse, y_pred_inverse

def plot_training_curves(history, save_path):
    """
    Plot training and validation loss and MAE curves.

    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit().
    """
    
    # Plot training & validation loss values
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    
def plot_predictions_and_truth(y_test_inverse, y_pred_inverse, save_path):
    """Plot the true values and predictions over time.

    Args:
        y_test_inverse (np.ndarray): Inverse transformed true values.
        y_pred_inverse (np.ndarray): Inverse transformed predicted values.
    """
    time_steps = np.arange(len(y_test_inverse))

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot True Values
    axs[0].plot(time_steps, y_test_inverse, label='True Values', color='blue', linestyle='-', linewidth=2, marker='o', markersize=4)
    axs[0].set_title('True Values Over Time', fontsize=16, fontweight='bold')
    axs[0].set_ylabel('Value', fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(visible=True, linestyle='--', alpha=0.7)

    # Plot Predictions
    axs[1].plot(time_steps, y_pred_inverse, label='Predictions', color='red', linestyle='--', linewidth=2, marker='x', markersize=4)
    axs[1].set_title('Predictions Over Time', fontsize=16, fontweight='bold')
    axs[1].set_xlabel('Time Steps', fontsize=14)
    axs[1].set_ylabel('Value', fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(visible=True, linestyle='--', alpha=0.7)

    # Add a reference horizontal line at y=0
    for ax in axs:
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Improve spacing
    plt.tight_layout()
    plt.savefig(save_path)
    
def plot_pred_vs_true_over_time(y_test_inverse, y_pred_inverse, save_path):
    """
    Plot the predicted values against the true values over time.

    Args:
        y_test_inverse (np.ndarray): Inverse transformed true values.
        y_pred_inverse (np.ndarray): Inverse transformed predicted values.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_inverse, label='True Values', color='blue')
    plt.plot(y_pred_inverse, label='Predictions', color='red', linestyle="dashed")
    plt.title('Predictions vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    
    plt.savefig(save_path)