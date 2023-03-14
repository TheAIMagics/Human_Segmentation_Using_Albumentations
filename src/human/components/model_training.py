import os,sys
import torch
from src.human.logger import logging
from src.human.exception import CustomException
from src.human.entity.config_entity import *
from src.human.entity.artifact_entity import *
from src.human.entity.custom_model import SegmentationModel
from src.human.utils import *

class ModelTraining:
    def __init__(self, data_transformation_artifact : DataTransformationArtifacts,
    model_trainer_config : ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def load_to_GPU(self, training_dl, valid_dl, model):
        try:
            logging.info('loading model to GPU')
            DEVICE = get_default_device()

            model = to_device(model, DEVICE)

            logging.info('loading data to GPU')
            training_dl = DeviceDataLoader(training_dl, DEVICE)
            valid_dl = DeviceDataLoader(valid_dl, DEVICE)

            logging.info("loading data and model to GPU is done")
            return training_dl, valid_dl, model
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_trainer(self):
        try:
            os.makedirs(self.model_trainer_config.model_trainer_artifact_dir,exist_ok=True)

            train_loader = torch.load(self.data_transformation_artifact.trainloader_path)
            valid_loader  = torch.load(self.data_transformation_artifact.validloader_path)

            logging.info("load the model")
            model =  SegmentationModel()
            torch.cuda.empty_cache()

            training_dl, valid_dl, model = self.load_to_GPU(train_loader, valid_loader, model)

            optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
            
            best_valid_loss = np.Inf

            for i in range(EPOCHS):

                train_loss = train_fn(training_dl, model, optimizer)
                valid_loss = eval_fn(valid_dl, model)           

                if valid_loss < best_valid_loss:
                    torch.save(model.state_dict(), 'best_model.pt')
                    print("saved model")
                    best_valid_loss = valid_loss

                print(f"Epoch : {i+1} Train_loss : {train_loss} valid_loss : {valid_loss}")

            logging.info(f"saving the model at {self.model_trainer_config.model_path}")
            torch.save(model.state_dict(), self.model_trainer_config.model_path)

            model_trainer_artifact = ModelTrainerArtifacts(
                model_path=self.model_trainer_config.model_path
            )
            logging.info(f"modler trainer artifact {model_trainer_artifact}")
            logging.info("model training completed")
            
            return model_trainer_artifact
        except Exception as e:
                raise CustomException(e, sys)