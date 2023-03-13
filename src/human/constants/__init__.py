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
