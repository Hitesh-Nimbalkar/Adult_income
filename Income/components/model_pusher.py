            
            
import yaml           
import shutil
import os           
import sys 
from Income.logger import logging
from Income.exception import ApplicationException           
from Income.entity.config_entity import ModelEvaluationConfig
from Income.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from Income.utils.utils import load_pickle_object,save_object,save_pickle_object
from Income.constant.training_pipeline import *     
from Income.constant import *       

            
            
            
            
          
class ModelPusher:

    def __init__(self,model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise ApplicationException(e, sys)
      
            
            
            
    def initiate_model_pusher(self):
        try:
            # Selected model path
            model_path = self.model_eval_artifact.model
            logging.info(f" Model path : {model_path}")
            model = load_pickle_object(file_path=model_path)
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,'model.pkl')
            
            save_pickle_object(file_path=file_path, model=model)
            logging.info("Model saved.")
            
            # Model report
            model_name = self.model_eval_artifact.model_name
            f1_score = self.model_eval_artifact.F1_Score
            accuracy = self.model_eval_artifact.accuracy
            
            
            # Create a dictionary for the report
            report = {'Model': model_name, 'Accuracy': accuracy, 'F1_Score': f1_score}
            
            logging.info(str(report))
            
            # Save the report as a YAML file
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,'model_report.yaml')
            logging.info(f"Report Location: {file_path}")

            # Save the report as a YAML file
            with open(file_path, 'w') as file:
                yaml.dump(report, file)

            logging.info("Report saved as YAML file.")
            
            
        

            model_pusher_artifact = ModelPusherArtifact(message="Model Pushed succeessfully")
            return model_pusher_artifact
        except  Exception as e:
            raise ApplicationException(e, sys)
    
            
            
    def __del__(self):
        logging.info(f"\n{'*'*20} Model Pusher log completed {'*'*20}\n\n")
            
            
            
            
            
            
            
            
            
            
 