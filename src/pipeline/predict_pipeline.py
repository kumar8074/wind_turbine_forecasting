import os
import sys
import numpy as np
import tensorflow as tf

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.utils.prediction_utils import load_keras_model, load_object, prepare_sequence_from_user_input
from src.logger import logging
from src.exception import CustomException

class PredictionPipeline:
    def __init__(self):
        self.model_path="artifacts/trained_model.keras"
        self.preprocessor_path="artifacts/preprocessor.pkl"
        
    def predict(self, user_input_dict):
        try:
            logging.info("Started Prediction Pipeline")
            
            # Load Processor object and model
            processor=load_object(self.preprocessor_path)
            logging.info("Preprocessor object loaded sucessfully")
            
            model=load_keras_model(self.model_path)
            logging.info("Model loaded sucessfully")
            
            # Create test set from user input
            features=['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
            logging.info("Preparing sequence from user input...")
            sequence = prepare_sequence_from_user_input(
                user_input_dict=user_input_dict,
                csv_path="DATA/T1.csv",
                scaler=processor,
                features=features,
                time_step=10
            )
            logging.info(f"Prepared sequence shape: {sequence.shape}")
            
            logging.info("Starting prediction...")
            # Make predictions
            y_pred=model.predict(sequence)
            
            # Get the number of features
            num_features = sequence.shape[2]
            logging.info(f"Number of features in the test data: {num_features}")

            # Create a dummy input with all zeros
            dummy = np.zeros((1, num_features))
            
            # Put your prediction in the same index where your target was during training (assumed 0)
            dummy[0, 0] = y_pred.flatten()[0]
            # apply inverse transform
            y_pred_original = processor.inverse_transform(dummy)[0, 0]
            
            logging.info(f"Predicted LV ActivePower (kW) for the next 10th minute: {y_pred_original}")
            
            logging.info("Prediction pipeline completed successfully")
        except CustomException as ce:
            logging.error(f"An error occurred: {ce}")
        except Exception as e:
            logging.error(f"Unexpected error during prediction pipeline: {e}")
            print(f"Unexpected error during prediction pipeline: {e}")
            
# Example Usage:
if __name__ == "__main__":
    predict_pipeline=PredictionPipeline()
    user_input_example={
        'LV ActivePower (kW)': 318,
        'Wind Speed (m/s)': 5.31,
        'Theoretical_Power_Curve (KWh)': 416,
        'Wind Direction (°)': 259
    }
    predict_pipeline.predict(user_input_example)
    
