import os
from datetime import datetime
from dataclasses import dataclass


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

DATABASE_NAME = "projects_db"
DATASET_NAME = "visa_dataset"
MYSQL_ENGINE_URL = "MYSQL_ENGINE_URL"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "visa_dataset.csv"
MODEL_FILE_NAME = "model.pkl"



# Data Ingestion Configuration
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2