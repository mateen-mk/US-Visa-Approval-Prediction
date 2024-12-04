import os
import sys

import pickle
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.US_Visa_Approval.exception import USvisaException
from src.US_Visa_Approval.logger import logging



class TargetValueMapping:
    def __init__(self):
        self.Certified:int = 0
        self.Denied:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    



class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of UTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    


class LocalStorgeService:
    """
    This class mimics the behavior of S3 storage using a local folder.
    """

    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        os.makedirs(self.base_folder, exist_ok=True)

    def file_path_available(self, local_path: str) -> bool:
        return os.path.exists(os.path.join(self.base_folder, local_path))

    def save_model(self, local_path: str, model_object: USvisaModel):
        full_path = os.path.join(self.base_folder, local_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as file:
            pickle.dump(model_object, file)

    def load_model(self, local_path: str) -> USvisaModel:
        full_path = os.path.join(self.base_folder, local_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model not found at {full_path}")
        with open(full_path, 'rb') as file:
            return pickle.load(file)



class USvisaEstimator:
    """
    This class handles saving, loading, and predicting using a local folder.
    """

    def __init__(self, base_folder: str, model_path: str):
        self.storage = LocalStorgeService(base_folder=base_folder)
        self.model_path = model_path
        self.loaded_model: USvisaModel = None

    def is_model_present(self) -> bool:
        try:
            return self.storage.file_path_available(self.model_path)
        except Exception as e:
            raise USvisaException(e)

    def load_model(self) -> USvisaModel:
        try:
            return self.storage.load_model(self.model_path)
        except Exception as e:
            raise USvisaException(e)

    def save_model(self, model_object: USvisaModel):
        try:
            self.storage.save_model(self.model_path, model_object)
        except Exception as e:
            raise USvisaException(e)

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise USvisaException(e)