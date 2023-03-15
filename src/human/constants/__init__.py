import torch
import numpy as np

ARTIFACTS_DIR: str = "artifacts"
SOURCE_DIR_NAME: str = "src"
S3_BUCKET_DATA_URI = "s3://human-segmentation/data"
ZIP_FILE_NAME: str = "data.zip"

# constants related to data ingestion
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion"
S3_DATA_FOLDER_NAME: str = "data.zip"
UNZIPPED_FOLDER_NAME: str = "unzip"
DATA_DIR_NAME: str = "data"

# constants related to data tranformation
DATA_TRANSFORMATION_ARTIFACTS_DIR: str = "data_transformation"
CSV_FILE_NAME:str = "train.csv"
VALIDATION_CSV_FILE = "validation.csv"
SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0
IMAGE_SIZE = 320
BATCH_SIZE = 16
TEST_SIZE = 0.1
RANDOM_STATE = 42
DATALOADER = 'dataloader'
TRAIN_DATALOADER = 'train_data_loader.pt'
VALID_DATALOADER = 'valid_data_loader.pt'
TRANSFORM_OBJECT_NAME: str = "transform.pkl"
VALID_DIR = 'valid'

# constants related to model training
MODEL_TRAINER_ARTIFACTS_DIR : str = 'model_training'
MODEL_NAME: str = "model.pt"
TRANSFORM_OBJECT_NAME: str = "transform.pkl"
BATCH_SIZE: int = 15
EPOCHS: int = 1
LEARNING_RATE:float = 0.003
OPTIMIZER = torch.optim.RMSprop
ENCODER = 'timm-efficientnet-b0'
WIEGHTS = 'imagenet'
BEST_VALID_LOSS = np.Inf

# constants realted to model evaluation 
S3_BUCKET_MODEL_URI: str = "s3://human-segmentation/model/"
S3_BUCKET_VALIDATION_URI : str = "s3://human-segmentation/validation/"
MODEL_EVALUATION_DIR: str = "model_evaluation"
S3_MODEL_DIR_NAME: str = "s3_model"
S3_MODEL_NAME: str = "model.pt"
BASE_LOSS: int = 3


