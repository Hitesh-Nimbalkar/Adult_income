
from Income.exception import ApplicationException
from Income.logger import logging
import sys
import os
from Income.entity.config_entity import *
from Income.entity.artifact_entity import *
from Income.utils.utils import read_yaml,load_pickle_object
from Income.constant.training_pipeline import *

class ModelEvaluation:


    def __init__(self,
                    data_validation_artifact:DataValidationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:

            self.data_validation_artifact=data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise ApplicationException(e,sys)
        
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(" Model Evaluation Started ")
            ## Artifact trained Model  files
            model_trained_artifact_path = self.model_trainer_artifact.trained_model_object_file_path
            model_trained_report = self.model_trainer_artifact.model_artifact_report
            
            # Saved Model files
            saved_model_report_path = self.model_trainer_artifact.saved_model_report
            saved_model_path = self.model_trainer_artifact.saved_model_file_path

            
            
            logging.info(f" Artifact Trained model :")

            # Load the model evaluation report from the saved YAML file
            saved_model_report_data = read_yaml(file_path=saved_model_report_path)
            model_trained_report_data = read_yaml(file_path=model_trained_report)
            


            # Loading the models
            logging.info("Saved_models directory .....")
            os.makedirs(SAVED_MODEL_DIRECTORY,exist_ok=True)
            
            # Check if SAVED_MODEL_DIRECTORY is empty
            if not os.listdir(SAVED_MODEL_DIRECTORY):
                artifact_model_f1_score =float( model_trained_report_data['F1_Score'])
                artifact_model_accuracy = float(model_trained_report_data['Accuracy'])
                model_name = "Artifact Model"
                F1_Score = artifact_model_f1_score
                accuracy = artifact_model_accuracy
                model_path = model_trained_artifact_path
                model_report_path = model_trained_report
                
            else:
                saved_model = load_pickle_object(file_path=saved_model_path)
                artifact_model = load_pickle_object(file_path=model_trained_artifact_path)

                # Compare the F1_Scores and accuracy of the two models
                saved_model_f1_score = float(saved_model_report_data['F1_Score'])
                saved_model_accuracy = float(saved_model_report_data['Accuracy'])

                artifact_model_f1_score =float( model_trained_report_data['F1_Score'])
                artifact_model_accuracy = float(model_trained_report_data['Accuracy'])

                # Compare the models and log the result
                if artifact_model_f1_score > saved_model_f1_score:
                    logging.info("Trained model outperforms the saved model!")
                    model_path = model_trained_artifact_path
                    model_report_path = model_trained_report
                    model_name = "Trained Model"
                    F1_Score = artifact_model_f1_score
                    logging.info(f"F1_Score : {F1_Score}")
                    accuracy = artifact_model_accuracy
                    logging.info(f" Model Accuracy : {accuracy}")
                elif artifact_model_f1_score < saved_model_f1_score:
                    logging.info("Saved model outperforms the trained model!")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    model_name = "Saved Model"
                    F1_Score = saved_model_f1_score
                    accuracy = saved_model_accuracy
                    logging.info(f"F1_Score : {F1_Score}")
                    logging.info(f" Model Accuracy : {accuracy}")
                else:
                    logging.info("Both models have the same F1_Score.")
                    F1_Score = saved_model_f1_score
                    accuracy = saved_model_accuracy
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    model_name = "Saved Model"
                
                

            # Create a model evaluation artifact
            model_evaluation = ModelEvaluationArtifact(model_name=model_name, F1_Score=F1_Score, accuracy=accuracy,
                                                    model=model_path, model_report_path=model_report_path)

            logging.info("Model evaluation completed successfully!")

            return model_evaluation
        except Exception as e:
            logging.error("Error occurred during model evaluation!")
            raise ApplicationException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")