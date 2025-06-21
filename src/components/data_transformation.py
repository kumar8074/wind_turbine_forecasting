import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
  
from src.utils.ingestion_utils import load_config  
from src.utils.transformation_utils import create_sequences
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    train_size: float
    val_size: float
    test_size: float
    window_size: float
    processor_file_path: str
    test_data_path: str
    
class DataTransformation:
    def __init__(self):
        self.config=load_config()
        self.data_transformation_config= DataTransformationConfig(
            train_size=self.config['transformation']['train_size'],
            val_size=self.config['transformation']['val_size'],
            test_size=self.config['transformation']['test_size'],
            window_size=self.config['transformation']['window_size'],
            processor_file_path=self.config['transformation']['preprocessor_file_path'],
            test_data_path=self.config['transformation']['test_data_path']
        )
        
    def initiate_data_transformation(self, df):
        logging.info("Entered the data transformation component")
        try:
            # Convert the 'Date/Time' to datetime format
            df['Date/Time'] = pd.to_datetime(df['Date/Time'], format="%d %m %Y %H:%M", errors='coerce')
            
            # Set 'Date/Time' as the index
            df.set_index('Date/Time', inplace=True)
            
            # Save the features
            features= df.columns.tolist()
            
            # Convert the DataFrame to a Numpy array
            data=df.values
            
            # Split the data into training, validation, and test sets
            train_size = int(len(df) * self.data_transformation_config.train_size)
            val_size = int(len(df) * self.data_transformation_config.val_size)
            test_size = int(len(df) * self.data_transformation_config.test_size)
            
            train_data= data[:train_size]
            val_data= data[train_size:train_size + val_size]
            test_data= data[train_size + val_size:]
            logging.info("Splitted data into training, validation and test sets")
            
            # Standardize the data using MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_data_scaled = scaler.fit_transform(train_data)
            
            # Transform validation and test using the same scaler
            val_data_scaled = scaler.transform(val_data)
            test_data_scaled = scaler.transform(test_data)
            logging.info("Standardized the data")
            
            X_train, y_train = create_sequences(train_data_scaled, time_step=self.data_transformation_config.window_size)
            X_val, y_val = create_sequences(val_data_scaled, time_step=self.data_transformation_config.window_size)
            X_test, y_test = create_sequences(test_data_scaled, time_step=self.data_transformation_config.window_size)
            logging.info("Created X,y Pairs")
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Ensure that artifacts and data directory exists
            os.makedirs(os.path.dirname(self.data_transformation_config.processor_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.test_data_path), exist_ok=True)
            
            # Save the processor object for later usecase.
            joblib.dump(scaler, self.data_transformation_config.processor_file_path)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.processor_file_path}")
            
            # Save the test data
            np.savez(self.data_transformation_config.test_data_path, X_test=X_test, y_test=y_test)
            logging.info(f"Test data saved at {self.data_transformation_config.test_data_path}")
            
            return(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test, 
                features
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e)
        
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
    except CustomException as ce:
        logging.error(f"An error occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error during transformation: {e}")

            
            
    