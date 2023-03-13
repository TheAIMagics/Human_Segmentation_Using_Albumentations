from src.human.logger import logging
from src.human.exception import CustomException
from src.human.constants import *
from src.human.entity.config_entity import *
from src.human.entity.artifact_entity import *
from src.human.components.data_ingestion import DataIngestion
from src.human.components.data_transformation import DataTransformation

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

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
        
    def run_pipeline(self) -> None:
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifacts=data_ingestion_artifacts)
        except Exception as e:
            raise CustomException(e, sys)