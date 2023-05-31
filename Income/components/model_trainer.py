import ast
import logging
import sys
import time
import os
import yaml
from Income.utils.utils import read_yaml,load_pickle_object,save_object,load_numpy_array_data
    
from Income.configuration.configuration import *
from Income.entity.config_entity import *
from Income.entity.artifact_entity import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
from collections import namedtuple
    
    
    
class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                    data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"\n{'*'*20} Model Training started {'*'*20}\n\n")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.results = []
            
            ## Loading Numpy arrays 
            logging.info
            self.input_train_array=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_input_train_path,file_name=DATA_TRANSFORMATION_INPUT_TRAIN)
            self.input_test_array=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_input_test_path,file_name=DATA_TRANSFORMATION_INPUT_TEST)
            self.target_train_array=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_target_train_path,file_name=DATA_TRANSFORMATION_TARGET_TRAIN)
            self.target_test_array=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_target_test_path,file_name=DATA_TRANSFORMATION_TARGET_TEST)
            
            # Define the models and their hyperparameters
            # Create a list of classification models with hyperparameters
            self.models = [
                        # ('Logistic_Regression', LogisticRegression(), {'model__C': [0.1, 1, 10]}),
                         ('Decision_Tree', DecisionTreeClassifier(), {'model__max_depth': [None, 5, 10]}),
                         ('Random_Forest', RandomForestClassifier(), {'model__max_depth': [15, 20, 25, 30, 40], 'model__n_estimators': [150, 200]})

                        #('KNN', KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7]}),
                        # ('SVM', SVC(), {'model__C': [0.1, 1, 10]})
                        ]
            
            
            self.ModelResult = namedtuple('ModelResult', ['Model', 'Accuracy', 'F1_Score', 'Training_Time'])
        except Exception as e:
            raise ApplicationException(e, sys) 
    
    



    def perform_gridsearch_cv(self):
        try:
            logging.info("Loading preprocessor object...")
            preprocessor_path = self.data_transformation_artifact.preprocessed_object_file_path
            preprocessor = load_pickle_object(preprocessor_path)
            
            logging.info("Loaded object")

            # Create an empty list to store the results
            results = []

            # Iterate over the models and perform training, evaluation, and hyperparameter tuning
            for model_name, model, param_grid in self.models:
                start_time = time.time()

                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])

                # Perform Grid Search for hyperparameter tuning
                grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
                grid_search.fit(self.input_train_array, self.target_train_array)
                end_time = time.time()

                # Calculate accuracy and F1_Score on test set using best model
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(self.input_test_array)
                accuracy = accuracy_score(self.target_test_array, y_pred)
                f1 = f1_score(self.target_test_array, y_pred)

                # Calculate training time
                training_time = end_time - start_time

                # Append results to list
                results.append({'Model': model_name, 'Accuracy': accuracy, 'F1_Score': f1, 'Training Time': training_time})

            # Select the best model based on the F1_Score
            best_f1_model = max(results, key=lambda x: x['F1_Score'])

            # Log the results
            logging.info("Model Training Results:")
            for result in results:
                logging.info(f"Model: {result['Model']}")
                logging.info(f"Accuracy: {result['Accuracy']}")
                logging.info(f"F1_Score: {result['F1_Score']}")
                logging.info(f"Training Time: {result['Training Time']} seconds")

            return best_f1_model

        except Exception as e:
            raise ApplicationException(e, sys) from e

    def select_best_model(self, results):
        best_accuracy_model = max(results, key=lambda x: x['Accuracy'])
        best_f1_model = max(results, key=lambda x: x['F1_Score'])
        return best_accuracy_model, best_f1_model
    
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Grid search cv...")
            best_f1_model = self.perform_gridsearch_cv()

            logging.info("-----------------------")

            # Log the F1_Score and accuracy of the selected model
            selected_model_name = best_f1_model['Model']
            selected_model_f1_score = best_f1_model['F1_Score']
            selected_model_accuracy = best_f1_model['Accuracy']
            logging.info(f"Selected Model: {selected_model_name}")
            logging.info(f"F1_Score: {selected_model_f1_score}")
            logging.info(f"Accuracy: {selected_model_accuracy}")

            # Log a message with an emoji
            logging.info("Model training completed successfully! ")

            # Log a message with multiple emojis
            logging.info("Best model selected! ")
            
            
            ## Save model to artifact 
            trained_model_object_file_path = self.model_trainer_config.trained_model_file_path
            save_object(file_path=trained_model_object_file_path, obj=best_f1_model)
            # Convert parameter values to strings
            selected_model_name_str = str(selected_model_name)
            selected_model_f1_score_str = str(selected_model_f1_score)
            selected_model_accuracy_str = str(selected_model_accuracy)
            
            # Create a report
            report = {'Model Name': selected_model_name_str, 'F1_Score': selected_model_f1_score_str, 'Accuracy':selected_model_accuracy_str}
            
            # Save report in artifact folder
            model_artifact_report_path=self.model_trainer_config.model_artifact_report
            report_file_path =model_artifact_report_path  
            with open(report_file_path, 'w') as file:
                yaml.safe_dump(report, file)
            logging.info("-----------------------") 
            
            
            saved_model_file_path = self.model_trainer_config.saved_model_file_path
            saved_model_report_path=self.model_trainer_config.saved_model_report_path




            logging.info("-----------------------")
            
            logging.info(f"trained_model : {trained_model_object_file_path}")
            logging.info(f" Trained model report : {model_artifact_report_path}")
            logging.info(f" Saved Model file path : {saved_model_file_path}")
            logging.info(f" Saved Model Report path : {saved_model_report_path}")
            
            
            logging.info("Report created")
            model_trainer_artifact = ModelTrainerArtifact(is_trained=True,
                                                        message="Model Training Done!!",
                                                        trained_model_object_file_path=trained_model_object_file_path,
                                                        model_artifact_report=model_artifact_report_path,
                                                        saved_model_file_path=saved_model_file_path,
                                                        saved_model_report=saved_model_report_path
                                                        )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Model Training log completed {'*'*20}\n\n")