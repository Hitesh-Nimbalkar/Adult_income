import os
import logging
from Income.logger import logging
from Income.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Income.utils.utils import read_yaml
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
        
        
    def data_dump(self,filepath):
        df = pd.read_csv(filepath)
        print(f"Rows and columns: {df.shape}")

        # Convert dataframe to json so that we can dump these record in mongo db
        df.reset_index(drop=True,inplace=True)
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        json_record = list(json.loads(df.T.to_json()).values())
        print(json_record[0])
        
        print("Data Uploaded")

        # Check if the database exists
        if DATABASE_NAME in client.list_database_names():
            print(f"The database {DATABASE_NAME} already exists")
            # Check if the collection exists
            if BATCH_COLLECTION_PATH in client[DATABASE_NAME].list_collection_names():
                print(f"The collection {BATCH_COLLECTION_PATH} already exists")
                # Drop the existing collection
                client[DATABASE_NAME][COLLECTION_NAME].drop()
                print(f"The collection {BATCH_COLLECTION_PATH} is dropped and will be replaced with new data")
            else:
                print(f"The collection {BATCH_COLLECTION_PATH} does not exist and will be created")
        else:
            # Create the database and collection
            print(f"The database {DATABASE_NAME} does not exist and will be created")
            db = client[DATABASE_NAME]
            col = db[COLLECTION_NAME]
            print(f"The collection {BATCH_COLLECTION_PATH} is created")

        # Insert converted json record to mongo db
        client[DATABASE_NAME][BATCH_COLLECTION_PATH].insert_many(json_record)
        
        logging.info("Prediction Data Updated to MongoDB")
        
    
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
            with open(self.model_file_path, 'rb') as f:
                model = pickle.load(f)
                
            logging.info(f" Model File Path : {self.model_file_path}")
                
                        # Feature Labels 
            schema = read_yaml("config/schema.yaml")
            input_features = schema['numerical_columns']
            categorical_features = schema['categorical_columns']
            target_features = schema['target_column']
            drop_columns = schema['drop_columns']
            all_columns=input_features+categorical_features+target_features
                
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
            file_path = os.path.join(FEATURE_ENG, 'batch_fea_eng.csv')
            df.to_csv(file_path, index=False)
            logging.info("Feature-engineered batch data saved as CSV.")

        # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df.drop('income',axis=1))
            
            print(type(transformed_data))

            # Make predictions using the model
            predictions = model.predict(transformed_data)

            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])

            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION, 'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logging.info(f"Batch predictions saved to '{csv_path}'.")

        except Exception as e:
            logging.error("An error occurred during batch prediction.")
 





