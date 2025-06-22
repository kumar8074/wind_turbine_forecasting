import os
import sys

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def train(self):
        try:
            logging.info("Starting Training Pipeline")
            # Step 1: Data Ingestion
            df = self.data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed")

            # Step 2: Data Transformation
            X_train, y_train, X_val, y_val, X_test, y_test, features = self.data_transformation.initiate_data_transformation(df)
            logging.info("Data Transformation completed")
            
            # Step 3: Model Training
            history = self.model_trainer.initiate_model_training(X_train, y_train, X_val, y_val, X_test, y_test)
            logging.info("Training Pipeline completed")
            print("Training Pipeline completed")
        except CustomException as ce:
            logging.error(f"An error occurred: {ce}")
        except Exception as e:
            logging.error(f"Unexpected error during training pipeline: {e}")
            print(f"Unexpected error during training pipeline: {e}")

# Example usage:
#if __name__ == "__main__":
    #train_pipeline = TrainingPipeline()
    #train_pipeline.train()
    #print("Training Pipeline executed successfully.")
