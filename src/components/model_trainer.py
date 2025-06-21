import os
import sys
from dataclasses import dataclass
import tensorflow as tf

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.utils.ingestion_utils import load_config  
from src.utils.trainer_utils import save_keras_model
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.models.model import build_model

@dataclass
class ModelTrainerConfig:
    model_save_path: str
    epochs: int
    batch_size: int
    
class ModelTrainer:
    def __init__(self):
        self.config=load_config()
        self.model_trainer_config= ModelTrainerConfig(
            model_save_path=self.config['model']['model_save_path'],
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size']
        )
        
    def initiate_model_training(self, X_train, y_train, X_val, y_val):
        try:
            logging.info("Entered Model Trainer Component")
            
            # Load the model
            model=build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            logging.info("Model Loaded successfully")
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=self.model_trainer_config.epochs, batch_size=self.model_trainer_config.batch_size, validation_data=(X_val, y_val))
            logging.info("Model Training Completed")
            
            # Save the trained model locally
            save_keras_model(
                save_path=self.model_trainer_config.model_save_path,
                model=model
            )
            logging.info(f"Trained model saved at {self.model_trainer_config.model_save_path}")
            
            return history
        except Exception as e:
            raise CustomException(e, sys)
        
        
# Example Usage:
if __name__ == "__main__":
    
    try:
        # Step 1: Ingest the raw data
        data_ingestion = DataIngestion()
        df = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        
        # Step 2: Transform the data
        data_transformation = DataTransformation()
        X_train, y_train, X_val, y_val, X_test, y_test, features = data_transformation.initiate_data_transformation(df)
        logging.info("Data Transformation completed")
        print("Data transformation complete.")
        
        # Step 3: Train the model
        model_trainer= ModelTrainer()
        history=model_trainer.initiate_model_training(X_train, y_train, X_val, y_val)
        print("Model Training Completed")
    except CustomException as ce:
        logging.error(f"An error occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error during transformation: {e}")
            
    
