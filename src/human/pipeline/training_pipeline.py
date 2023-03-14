from src.human.logger import logging
from src.human.exception import CustomException
from src.human.constants import *
from src.human.entity.config_entity import *
from src.human.entity.artifact_entity import *
from src.human.components.data_ingestion import DataIngestion
from src.human.components.data_transformation import DataTransformation
from src.human.components.model_training import ModelTraining

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try: 
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion step completed successfully in train pipeline")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transformation(self,data_ingestion_artifacts: DataIngestionArtifacts) ->DataTransformationArtifacts:
        logging.info("Starting data preprocessing in training pipeline")
        try: 
            data_transformation = DataTransformation(data_transformation_config=self.data_transformation_config,
            data_ingestion_artifact=data_ingestion_artifacts)
            data_preprocessing_artifacts = data_transformation.initiate_data_transformation()
            logging.info("Data preprocessing step completed successfully in train pipeline")
            return data_preprocessing_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self,data_transformation_artifact : DataTransformationArtifacts):
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            
            model_trainer = ModelTraining(data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=self.model_trainer_config)
            model_trainer_artifact =  model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_pipeline(self) -> None:
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifacts=data_ingestion_artifacts)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        except Exception as e:
            raise CustomException(e, sys)