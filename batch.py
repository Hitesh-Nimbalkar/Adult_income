import os
import logging
from Income.logger import logging
from Income.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Income.utils.utils import read_yaml,load_pickle_object,save_array_to_directory,load_numpy_array_data
from Income.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
import sys 
import pymongo
import json
from Income.constant.data_base import *
from Income.constant import *
import urllib
import yaml
import numpy as np



env_file_path = os.path.join(ROOT_DIR, 'env.yaml')

# Load environment variables from env.yaml
with open(env_file_path) as file:
    env_vars = yaml.safe_load(file)
username = env_vars.get('USER_NAME')
password = env_vars.get('PASS_WORD')

# Use the escaped username and password in the MongoDB connection string
mongo_db_url = f"mongodb+srv://{username}:{password}@rentalbike.5fi8zs7.mongodb.net/"

client = pymongo.MongoClient(mongo_db_url)




class batch_prediction:
    def __init__(self,input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

        
    
    def start_batch_prediction(self):
        try:
            logging.info("Loading the saved pipeline")

            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            # Load the model separately
            model =load_pickle_object(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")

            # Feature Labels
            schema = read_yaml("config/schema.yaml")
            input_features = schema['numerical_columns']
            categorical_features = schema['categorical_columns']
            target_features = schema['target_column']
            drop_columns = schema['drop_columns']
            all_columns = input_features + categorical_features + target_features

            # Create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])

            # Read the input file
            df = pd.read_csv(self.input_file_path)

            # Apply feature engineering
            array = feature_engineering_pipeline.transform(df)
            df = pd.DataFrame(array, columns=all_columns)

            # Save the feature-engineered data as a CSV file
            FEATURE_ENG_PATH = FEATURE_ENG  # Specify the desired path for saving the CSV file
            os.makedirs(FEATURE_ENG_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENG_PATH, 'batch_fea_eng.csv')
            df.to_csv(file_path, index=False)
            logging.info("Feature-engineered batch data saved as CSV.")
            df=df.drop('income', axis=1)
                
                
            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")
            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)
            logging.info(f"Transformed Data Shape: {transformed_data.shape}")
            
            
            save_array_to_directory(directory_path=FEATURE_ENG_PATH, array=transformed_data,file_name='batch')

            logging.info("Saved array to Directory")
            
            # Make predictions using the model
            print(type(model))
            array=load_numpy_array_data(file_path=FEATURE_ENG_PATH,file_name='batch')
            
            logging.info(f"Loaded numpy from batch prediciton :{array}")
            predictions = model.predict(array)
            logging.info("Predictions done")

            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])

            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH, 'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logging.info(f"Batch predictions saved to '{csv_path}'.")

        except Exception as e:
            ApplicationException(e,sys) 



