import sys
from Income.exception import ApplicationException
from Income.logger import logging
from Income.configuration.configuration import Configuration
from Income.components.data_ingestion import DataIngestion
from Income.components.data_validation import DataValidation
from Income.components.data_transformation import DataTransformation
from Income.components.model_trainer import ModelTrainer
from Income.components.model_evaluation import ModelEvaluation
from Income.components.model_pusher import ModelPusher
from Income.entity.artifact_entity import *
from Income.entity.config_entity import *



class Pipeline():
    def __init__(self, config:Configuration) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ApplicationException(e, sys) from e


    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,
                                       data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config = self.config.get_data_transformation_config(),
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact = data_validation_artifact)

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
        
        
    def start_model_training(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                        data_transformation_artifact=data_transformation_artifact)   

            return model_trainer.initiate_model_training()
        except Exception as e:
            raise ApplicationException(e,sys) from e  
        
    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
    
            model_eval = ModelEvaluation(data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise  ApplicationException(e,sys)
        
        
    
    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                 model_trainer_artifact:ModelTrainerArtifact,
                                ):
        try:
            model_eval = ModelEvaluation(data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except  Exception as e:
            raise  ApplicationException(e,sys)
        
    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher = ModelPusher(model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise  ApplicationException(e,sys)
        
        
        
        
        
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                    data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)  
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
         
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
train=Pipeline(config=Configuration())
train.run_pipeline()