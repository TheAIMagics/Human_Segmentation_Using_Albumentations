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
SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0
IMAGE_SIZE = 320
BATCH_SIZE = 16
TEST_SIZE = 0.2
RANDOM_STATE = 42
DATALOADER = 'dataloader'
TRAIN_DATALOADER = 'train_data_loader.pt'
VALID_DATALOADER = 'valid_data_loader.pt'
TRANSFORM_OBJECT_NAME: str = "transform.pkl"