import sys
import os
import shutil
from src.US_Visa_Approval.exception import USvisaException
from src.US_Visa_Approval.logger import logging
from src.US_Visa_Approval.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.US_Visa_Approval.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def save_model_locally(self, source_path: str, destination_path: str) -> None:
        """
        Copies the trained model from the source path to the destination path.

        :param source_path: Path of the trained model file
        :param destination_path: Target path to save the model
        """
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(source_path, destination_path)
            logging.info(f"Model saved locally at: {destination_path}")
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function saves the trained model to a local directory.

        Output      :   Returns a model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Saving trained model to local folder")

            # Save the model locally
            self.save_model_locally(
                source_path=self.model_evaluation_artifact.trained_model_path,
                destination_path=self.model_pusher_config.model_pusher_path
            )

            # Create a ModelPusherArtifact
            model_pusher_artifact = ModelPusherArtifact(
                model_pusher_path=self.model_pusher_config.model_pusher_path
            )

            logging.info("Model saved to local folder successfully")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
