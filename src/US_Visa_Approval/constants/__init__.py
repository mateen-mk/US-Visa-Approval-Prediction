import os
from datetime import datetime, date
from dataclasses import dataclass



TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


DATABASE_NAME = "projects_db"
MYSQL_ENGINE_URL = "MYSQL_ENGINE_URL"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# dataset name for saving it in 'artifact/data' after importing it from MySQL
FILE_NAME: str = "visa_dataset.csv"
MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")



'''
Training related Constant
'''

# Data Ingestion related constants
DATASET_NAME = "visa_dataset"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_DATA_DIR: str = "data"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# Data Transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"