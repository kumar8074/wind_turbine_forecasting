import pandas as pd
import os
import sys
from dataclasses import dataclass

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.ingestion_utils import load_config
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    
class DataIngestion:
    def __init__(self):
        self.config=load_config()
        self.ingestion_config= DataIngestionConfig(raw_data_path=self.config['raw_data_path'])
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df=pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Loaded data from CSV file")
            return df
        except FileNotFoundError:
            logging.error(f"File not found at: {self.ingestion_config.raw_data_path}")
            raise CustomException(f"File not found at: {self.ingestion_config.raw_data_path}")
        except Exception as e:
            raise CustomException(e)
        
        
# Example usage:
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    try:
        data = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        logging.info(data.head())
        print(data.head())
    except CustomException as e:
        logging.error(f"An error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")