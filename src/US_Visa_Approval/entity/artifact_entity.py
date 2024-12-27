from dataclasses import dataclass



# Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    trained_file_path:str   # file path to train.csv file in the process of data ingestion
    test_file_path:str      # file path to test.csv file in the process of data ingestion


# Data Validation Artifact
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    drift_report_file_path: str


# Data Transfomation Artifact
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str    # file path to preprocessing.pkl
    transformed_train_file_path:str     # file path to trained data in numpy array format (train.npy)
    transformed_test_file_path:str      # file path to test data in numpy array format (test.npy)


# Classification Matrix Artifact
@dataclass
class ClassificationMetrixArtifact:
    f1_score:float
    precision_score:float
    recall_score:float


# Model Training Artifacts
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetrixArtifact


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str


@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str 
