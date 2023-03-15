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

# Model trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    model_path: str

# Model Evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    s3_model_loss: float
    is_model_accepted: bool
    trained_model_path: str
    s3_model_path: str

# Model Pusher artifacts
@dataclass
class ModelPusherArtifacts:
    response: dict