import os,sys
import torch
import joblib
import numpy as np
from src.human.utils import *
from src.human.logger import logging
from src.human.exception import CustomException
from src.human.entity.config_entity import *
from src.human.entity.artifact_entity import *
from src.human.entity.custom_model import SegmentationModel
from src.human.cloud_storage.s3_opearations import S3Sync

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, 
    data_transformation_artifact=DataTransformationArtifacts,
    model_trainer_artifacts = ModelTrainerArtifacts) -> None:
        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifacts = model_trainer_artifacts

    def get_best_model_path(self):
        try:
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            os.makedirs(os.path.dirname(best_model_dir), exist_ok=True)
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(
                folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pt"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info("Model is not available in best_model_directory")

            return best_model_path

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_model(self):
        try:
            transformer_object = joblib.load(self.data_transformation_artifact.transformer_object_path)
            test_dl =  torch.load(self.data_transformation_artifact.validloader_path)
            best_model_path = self.get_best_model_path()
            if best_model_path is not None:
                # load back the model
                logging.info("load the model")
                model =  SegmentationModel()
                torch.cuda.empty_cache()
                logging.info("load production model to GPU")
                DEVICE = get_default_device()
                model = to_device(model, DEVICE)
                model.load_state_dict(torch.load(self.model_trainer_artifacts.model_path))
                model.eval()
                logging.info(f"load the data to {DEVICE}")
                test_dl = DeviceDataLoader(test_dl, DEVICE)
                logging.info("evaluate production model on test data")
                s3_model_loss = eval_fn(test_dl, model)
            else:
                logging.info("Model is not found on production server, So couldn't evaluate")
                s3_model_loss = None

            return s3_model_loss

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        try:
            test_dl =  torch.load(self.data_transformation_artifact.validloader_path)
            s3_model_loss = self.evaluate_model()
            print("s3_model_loss",s3_model_loss)
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss
            model = SegmentationModel()
            torch.cuda.empty_cache()
            logging.info("load production model to GPU")
            DEVICE = get_default_device()
            model = to_device(model, DEVICE)
            model.load_state_dict(torch.load(self.model_trainer_artifacts.model_path))
            model.eval()
            logging.info(f"load the data to {DEVICE}")
            test_dl = DeviceDataLoader(test_dl, DEVICE)
            trained_model_loss = eval_fn(test_dl, model)
            print(trained_model_loss, BASE_LOSS)
            print(trained_model_loss < BASE_LOSS)
            evaluation_response = tmp_best_model_loss > trained_model_loss and trained_model_loss < BASE_LOSS
            model_evaluation_artifacts = ModelEvaluationArtifacts(
                s3_model_loss=tmp_best_model_loss,
                is_model_accepted=evaluation_response,
                trained_model_path=os.path.dirname(
                self.model_trainer_artifacts.model_path),
                s3_model_path=self.model_evaluation_config.s3_model_path
            )
            logging.info(f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")
            print("model_evaluation_artifacts",model_evaluation_artifacts)
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
