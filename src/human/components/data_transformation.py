import os
import sys
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt 
import albumentations as A
from pathlib import Path
from src.human.constants import *
from src.human.logger import logging
from src.human.exception import CustomException
from src.human.entity.config_entity import *
from src.human.entity.artifact_entity import *
from src.human.components.custom_dataset import CustomData
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
     data_ingestion_artifact: DataIngestionArtifacts)-> None:
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_train_augmentations(self):
        return A.Compose([
                            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                            A.HorizontalFlip(p = 0.5),
                            A.VerticalFlip(p = 0.5)
        ])

    def get_valid_augmentations(self):
        return A.Compose([
                            A.Resize(IMAGE_SIZE, IMAGE_SIZE)
        ])
        
    def initiate_data_transformation(self) ->DataTransformationArtifacts:
            try:
                logging.info("Initiating the data transformation component...")
                csv_file_path :str = os.path.join(self.data_ingestion_artifact.data_folder_path, CSV_FILE_NAME)
                dataframe = pd.read_csv(csv_file_path)
                train_df, valid_df = train_test_split(dataframe, test_size = TEST_SIZE, random_state = RANDOM_STATE)
                
                trainset = CustomData(train_df, self.get_train_augmentations(),data_ingestion_artifact=self.data_ingestion_artifact)
                validset = CustomData(valid_df, self.get_valid_augmentations(),data_ingestion_artifact=self.data_ingestion_artifact)
                logging.info(f"Size of Trainset : {len(trainset)}")
                logging.info(f"Size of Validset : {len(validset)}")

                os.makedirs(self.data_transformation_config.dataloader_path,exist_ok=True)

                logging.info("Saving transformer oblect for prediction")
                joblib.dump(self.get_train_augmentations(), self.data_transformation_config.transformer_object_path)

                trainloader = DataLoader(trainset, batch_size= BATCH_SIZE, shuffle= True)
                validloader = DataLoader(validset, batch_size = BATCH_SIZE)

                

                torch.save(trainloader, self.data_transformation_config.train_dataloader_path)
                torch.save(validloader, self.data_transformation_config.valid_dataloader_path)

                data_transformation_artifact = DataTransformationArtifacts(
                    trainloader_path=self.data_transformation_config.train_dataloader_path,
                    validloader_path=self.data_transformation_config.valid_dataloader_path,
                    transformer_object_path= self.data_transformation_config.transformer_object_path
                    )

                logging.info('Data transformation is completed Successfully.')
            except Exception as e:
                raise CustomException(e, sys)