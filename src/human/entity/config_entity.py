import os, sys
from dataclasses import dataclass
from from_root import from_root
from src.human.constants import *

@dataclass
class DataIngestionConfig:
    data_ingestion_artifact_dir: str = os.path.join(from_root(), ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
    download_dir = os.path.join(data_ingestion_artifact_dir, DATA_DIR_NAME)
    zip_data_path : str = os.path.join(download_dir, S3_DATA_FOLDER_NAME)
    unzip_data_dir: str = os.path.join(data_ingestion_artifact_dir, UNZIPPED_FOLDER_NAME,DATA_DIR_NAME)