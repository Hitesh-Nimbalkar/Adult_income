import os
import sys
from six.moves import urllib
import numpy as np
import pandas as pd
from Income.logger import logging
from Income.exception import ApplicationException
from Income.constant import *
from Income.constant.data_base import *
from Income.data_access.Mondo_data import mongodata
from Income.entity.config_entity import DataIngestionConfig
from Income.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*30}Data Ingestion log started.{'<<'*30} \n\n")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def get_data_from_mongo_DB(self) -> str:
        try:
            
            # Raw Data Directory Path
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            # Make Raw data Directory
            os.makedirs(raw_data_dir, exist_ok=True)

            file_name = FILE_NAME

            raw_file_path = os.path.join(raw_data_dir, file_name)

            logging.info(
                f"Downloading file from Mongo DB into :[{raw_file_path}]")
            
            data=mongodata()
            # Storing mongo data ccs file to raw directoy 
            data.export_collection_as_csv(collection_name=COLLECTION_NAME,database_name=DATABASE_NAME,file_path=raw_file_path)
            
            logging.info(
                f"File :[{raw_file_path}] has been downloaded successfully.")
            return raw_file_path

        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
        
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            file_name = os.listdir(raw_data_dir)[0]

            file_path = os.path.join(raw_data_dir, file_name)

            logging.info(f"Reading csv file: [{file_path}]")

            df  = pd.read_csv(file_path)

            logging.info(f"Splitting data into train and test")

            train_set = None
            test_set = None

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                          file_name)
# ***********************************************************************************************
            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                train_set.to_csv(train_file_path, index=False)

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed successfully."
                                                            )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise ApplicationException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            raw_file_path = self.get_data_from_mongo_DB()
            return self.split_data_as_train_test()
        except Exception as e:
            raise ApplicationException(e, sys)from e