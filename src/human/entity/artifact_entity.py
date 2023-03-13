from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    data_folder_path: str


# Data transformation artifacts
@dataclass
class DataTransformationArtifacts:
    trainloader_path :str
    validloader_path :str
    transformer_object_path : str
