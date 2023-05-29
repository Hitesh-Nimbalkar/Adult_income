import sys
from Income.exception import ApplicationException
from Income.logger import logging
from Income.configuration import configuration
from Income.components.data_ingestion import DataIngestion
from Income.entity.artifact_entity import *



class Pipeline():
    def __init__(self, config: configuration=configuration()) -> None:
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
        
        
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
         
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
train=Pipeline()
train.run_pipeline(configuration)