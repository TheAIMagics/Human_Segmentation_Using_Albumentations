import os,sys
from src.human.cloud_storage.s3_opearations import S3Sync
from src.human.logger import logging
from src.human.exception import CustomException
from src.human.constants import *
from src.human.entity.artifact_entity import *

class ModelPusher:
    def __init__(self, model_evaluation_artifacts: ModelEvaluationArtifacts, 
                 data_transformation_artifact : DataTransformationArtifacts):
        self.model_evaluation_artifacts = model_evaluation_artifacts
        self.data_transformation_artifact = data_transformation_artifact
    
    def initiate_model_pusher(self):
        try:
            logging.info("Initiating model pusher component")
            if self.model_evaluation_artifacts.is_model_accepted:
                trained_model_path = self.model_evaluation_artifacts.trained_model_path
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                s3_sync = S3Sync()
                s3_sync.sync_folder_to_s3(folder=trained_model_path, aws_bucket_url=s3_model_folder_path)
                s3_sync.sync_folder_to_s3(folder=self.data_transformation_artifact.validation_csv_file,
                                          aws_bucket_url=S3_BUCKET_VALIDATION_URI)
                message = "Model Pusher pushed the current Trained model to Production server storage"
                response = {"is model pushed": True, "S3_model": s3_model_folder_path + "/" + str(MODEL_NAME),"message" : message}
                logging.info(response)
            else:
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                message = "Current Trained Model is not accepted as model in Production has better loss"
                response = {"is model pushed": False, "S3_model":s3_model_folder_path,"message" : message}
                logging.info(response)
            model_pusher_artifacts = ModelPusherArtifacts(response=response)
            logging.info(f"Model evaluation completed! Artifacts: {model_pusher_artifacts}")
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys)