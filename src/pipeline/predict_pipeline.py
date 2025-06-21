import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.utils.prediction_utils import load_keras_model, load_object, calculate_r2_score
from src.logger import logging
from src.exception import CustomException

class PredictionPipeline:
    def __init__(self):
        self.test_data_path="DATA/test.csv.npz"
        self.model_path="artifacts/trained_model.keras"
        self.preprocessor_path="artifacts/preprocessor.pkl"
        
    def predict(self):
        try:
            logging.info("Started Prediction Pipeline")
            
            # Load the test data
            data=np.load(self.test_data_path)
            X_test=data['X_test']
            y_test=data['y_test']
            logging.info("Test data loaded successfully")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Load Processor object and model
            processor=load_object(self.preprocessor_path)
            logging.info("Preprocessor object loaded sucessfully")
            
            model=load_keras_model(self.model_path)
            logging.info("Model loaded sucessfully")
            
            logging.info("Starting prediction...")
            # Make predictions
            y_pred=model.predict(X_test)
            
            # Get the number of features
            num_features = X_test.shape[2]
            logging.info(f"Number of features in the test data: {num_features}")
            print(f"Number of features in the test data: {num_features}")
            
            # Calculate R2 Score
            logging.info("Calculating R2 Score...")
            r2 = calculate_r2_score(y_test, y_pred, num_features, processor)
            logging.info(f"Calculated R2 Score: {r2}")
            print(f"Calculated R2 Score: {r2}")
            logging.info("Prediction pipeline completed successfully")
        except CustomException as ce:
            logging.error(f"An error occurred: {ce}")
        except Exception as e:
            logging.error(f"Unexpected error during prediction pipeline: {e}")
            print(f"Unexpected error during prediction pipeline: {e}")
            
# Example Usage:
if __name__ == "__main__":
    predict_pipeline=PredictionPipeline()
    predict_pipeline.predict()