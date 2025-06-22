import os
import sys
from dataclasses import dataclass
import tensorflow as tf
import mlflow
from mlflow.models import infer_signature

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.utils.ingestion_utils import load_config  
from src.utils.trainer_utils import (save_keras_model, calculate_r2_score, 
                                     plot_training_curves, plot_predictions_and_truth,
                                     plot_pred_vs_true_over_time)

from src.utils.prediction_utils import load_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.models.model import build_model

@dataclass
class ModelTrainerConfig:
    model_save_path: str
    preprocessor_path: str
    visualizations_dir: str
    epochs: int
    batch_size: int
    tracking_uri: str
    experiment_name: str
    
class ModelTrainer:
    def __init__(self):
        self.config=load_config()
        self.model_trainer_config= ModelTrainerConfig(
            model_save_path=self.config['model']['model_save_path'],
            preprocessor_path=self.config['transformation']['preprocessor_file_path'],
            visualizations_dir=self.config['model']['visualizations_dir'],
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            tracking_uri=self.config['mlflow']['tracking_uri'],
            experiment_name=self.config['mlflow']['experiment_name']
        )
        
    def initiate_model_training(self, X_train, y_train, X_val, y_val, X_test, y_test):
        try:
            logging.info("Entered Model Trainer Component")
            
            mlflow.set_tracking_uri(uri=self.model_trainer_config.tracking_uri)
            mlflow.set_experiment(experiment_name=self.model_trainer_config.experiment_name)
            logging.info("MLflow tracking URI and experiment set successfully")
            
            # Load the preprocessor object
            processor=load_object(self.model_trainer_config.preprocessor_path)
            logging.info("Preprocessor object loaded successfully")
            
            # Load the model
            model=build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            logging.info("Model Built successfully")
            
            # Ensure the visualizations directory exists
            os.makedirs(self.model_trainer_config.visualizations_dir, exist_ok=True)
            logging.info(f"Visualizations directory created at {self.model_trainer_config.visualizations_dir}")
            
            with mlflow.start_run():
                
                # Log model config
                config = model.get_config()
                mlflow.log_param("num_layers", len(config["layers"]))
                mlflow.log_param("model_type", type(model).__name__)
                mlflow.log_param("input_shape", str(model.input_shape))
                mlflow.log_param("output_shape", str(model.output_shape))
                mlflow.log_param("epochs", self.model_trainer_config.epochs)
                mlflow.log_param("batch_size", self.model_trainer_config.batch_size)
                logging.info("Model parameters logged successfully")
                
                # Log optimizer details
                optimizer_config = model.optimizer.get_config()
                mlflow.log_param("optimizer", optimizer_config["name"])
                mlflow.log_param("learning_rate", optimizer_config["learning_rate"])
                
                # Train the model
                history = model.fit(X_train, y_train, epochs=self.model_trainer_config.epochs, batch_size=self.model_trainer_config.batch_size, validation_data=(X_val, y_val))
                logging.info("Model Training Completed")
                
                # Plot training curves
                logging.info("Plotting training curves...")
                plot_training_curves(history, save_path=self.model_trainer_config.visualizations_dir+ "/training_curves.png")
                logging.info("Training curves plotted successfully")
                
                # Log final metrics
                mlflow.log_metric("val_mae", history.history['val_mae'][-1])
                mlflow.log_metric("val_loss", history.history['val_loss'][-1])
                
                # make predictions on the test set
                y_pred = model.predict(X_test)
                num_features = X_test.shape[2]
                logging.info(f"Number of features in the test data: {num_features}")
                
                # Calculate R2 score
                logging.info("Calculating R2 Score...")
                r2_score, y_test_inverse, y_pred_inverse= calculate_r2_score(y_test, y_pred,num_features, processor)
                logging.info(f"Calculated R2 Score: {r2_score}")
                mlflow.log_metric("r2_score", r2_score)
                
                # Plot predictions and truth curves
                logging.info("Plotting Predictions and Truth curves...")
                plot_predictions_and_truth(y_test_inverse, y_pred_inverse, 
                    save_path=self.model_trainer_config.visualizations_dir + "/predictions_and_truth.png"
                )
                logging.info("Predictions and Truth curves plotted successfully")
                
                # Plot predicted vs true over time
                logging.info("Plotting Predicted vs True over time...")
                plot_pred_vs_true_over_time(y_test_inverse, y_pred_inverse, 
                    save_path=self.model_trainer_config.visualizations_dir + "/pred_vs_true_over_time.png"
                )
                logging.info("Predicted vs True over time plotted successfully")
                
                # Log the visualizations to MLflow
                mlflow.log_artifact(self.model_trainer_config.visualizations_dir+"/training_curves.png", artifact_path="visualizations")
                mlflow.log_artifact(self.model_trainer_config.visualizations_dir + "/predictions_and_truth.png", artifact_path="visualizations")
                mlflow.log_artifact(self.model_trainer_config.visualizations_dir + "/pred_vs_true_over_time.png", artifact_path="visualizations")
                
                # Infer the signature of the model
                signature = infer_signature(X_train, model.predict(X_train))
                logging.info("Model signature inferred successfully")
                
                # Log the model
                mlflow.tensorflow.log_model(
                    model=model,
                    name="artifacts",
                    signature=signature,
                )
                logging.info("Model logged to MLflow successfully")
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
        history=model_trainer.initiate_model_training(X_train, y_train, X_val, y_val, X_test, y_test)
        print("Model Training Completed")
    except CustomException as ce:
        logging.error(f"An error occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error during transformation: {e}")
            
    
