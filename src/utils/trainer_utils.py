import os
from tensorflow.keras.models import Model

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