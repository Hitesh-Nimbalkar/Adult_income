import os  
import sys 
from Income.configuration import *
from Income.entity.config_entity import DataIngestionConfig,DataValidationConfig
from Income.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from Income.exception import ApplicationException
from Income.logger import logging
from Income.utils.utils import read_yaml
from Income.entity.row_data_validation import IngestedDataValidation
import shutil
from Income.constant import *
from scipy.stats import ks_2samp
import pandas as pd
import json

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n") 
            
            # Creating_instance           
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            
            # Schema_file_path
            self.schema_path = self.data_validation_config.schema_file_path
            
            
            # creating instance for row_data_validation
            self.train_data = IngestedDataValidation(
                                validate_path=self.data_ingestion_artifact.train_file_path, schema_path=self.schema_path)
            self.test_data = IngestedDataValidation(
                                validate_path=self.data_ingestion_artifact.test_file_path, schema_path=self.schema_path)
            
            # Data_ingestion_artifact--->Unvalidated train and test file path
            self.train_path = self.data_ingestion_artifact.train_file_path
            self.test_path = self.data_ingestion_artifact.test_file_path
            
            # Data_validation_config --> file paths to save validated_data
            self.validated_train_path = self.data_validation_config.validated_train_path
            self.validated_test_path =self.data_validation_config.validated_test_path
            
        
        except Exception as e:
            raise ApplicationException(e,sys) from e


    def isFolderPathAvailable(self) -> bool:
        try:

            # check is the train and test file exists (Unvalidated file)
             
            isfolder_available = False
            train_path = self.train_path
            test_path = self.test_path
            if os.path.exists(train_path):
                if os.path.exists(test_path):
                    isfolder_available = True
            return isfolder_available
        except Exception as e:
            raise ApplicationException(e, sys) from e     
      


        
    def is_Validation_successfull(self):
        try:
            validation_status = True
            logging.info("Validation Process Started")
            if self.isFolderPathAvailable() == True:
                
                # Train file 
                train_filename = os.path.basename(
                    self.data_ingestion_artifact.train_file_path)

                is_train_filename_validated = self.train_data.validate_filename(
                    file_name=train_filename)

                is_train_column_name_same = self.train_data.check_column_names()
                validating_train_data_types=self.train_data.validate_data_types()

                is_train_missing_values_whole_column = self.train_data.missing_values_whole_column()
                
                
                self.train_data.replace_null_values_with_null()

                
                # Test File 
                test_filename = os.path.basename(
                    self.data_ingestion_artifact.test_file_path)

                is_test_filename_validated = self.test_data.validate_filename(
                    file_name=test_filename)

                is_test_column_name_same = self.test_data.check_column_names()
                validating_test_data_types=self.test_data.validate_data_types()

                is_test_missing_values_whole_column = self.test_data.missing_values_whole_column()

                self.test_data.replace_null_values_with_null()
                
                
                

                
                logging.info(
                    f"Train_set status: "
                    f"is Train filename validated? {is_train_filename_validated} | "
                    f"is train column name validated? {is_train_column_name_same} | "
                    f"whole missing columns? {is_train_missing_values_whole_column}"
                    f"Data type validation? {validating_train_data_types}"
                )
                logging.info(
                    f"Test_set status: "
                    f"is Test filename validated? {is_test_filename_validated} | "
                    f"is test column names validated? {is_test_column_name_same} | "
                    f"whole missing columns? {is_test_missing_values_whole_column}"
                    f"Data type validation? {validating_test_data_types}"
)
                if is_train_filename_validated  & is_train_column_name_same & is_train_missing_values_whole_column & validating_train_data_types :
                    
                    ## Exporting Train.csv file 
                    # Create the directory if it doesn't exist
                    os.makedirs(self.validated_train_path, exist_ok=True)

                    # Copy the CSV file to the validated train path
                    shutil.copy(self.train_path, self.validated_train_path)
                    self.validated_train_path=os.path.join(self.validated_train_path,FILE_NAME)
                    # Log the export of the validated train dataset
                    logging.info(f"Exported validated train dataset to file: [{self.validated_train_path}]")
                                     
                                     
                if is_test_filename_validated  & is_test_column_name_same & is_test_missing_values_whole_column & validating_test_data_types :
                                          
                    ## Exporting test.csv file
                    os.makedirs(self.validated_test_path, exist_ok=True)
                    logging.info(f"Exporting validated test dataset to file: [{self.validated_test_path}]")
                    os.makedirs(self.validated_test_path, exist_ok=True)
                    # Copy the CSV file to the validated train path
                    shutil.copy(self.test_path, self.validated_test_path)
                    self.validated_test_path=os.path.join(self.validated_test_path,FILE_NAME)
                    # Log the export of the validated train dataset
                    logging.info(f"Exported validated train dataset to file: [{self.validated_test_path}]")
                                        
                    
                    return validation_status,self.validated_train_path,self.validated_test_path
                else:
                    validation_status = False
                    logging.info("Check your Training Data! Validation Failed")
                    raise ValueError(
                        "Check your Training data! Validation failed")
                

            return validation_status,"NONE","NONE"
        except Exception as e:
            raise ApplicationException(e, sys) from e      
        
    
    def get_and_save_data_drift_report(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("Generating data drift report.json file")
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(train_df, test_df)
            report_file_path = self.data_validation_config.drift_report
            print(f"{report_file_path}")
            report = json.loads(profile.json())

            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
            logging.info("Report.json file generation successful!!")
            return report
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def save_data_drift_report_page(self,train_df,test_df):
        try:
            logging.info("Generating data drift report.html page")
            dashboard = Dashboard(tabs = [DataDriftTab()])
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.drift_report_page
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
            logging.info("Report.html page generation successful!!")
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            logging.info("Checking for Data Drift")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            report = self.get_and_save_data_drift_report(train_df=train_df,test_df=test_df)
            self.save_data_drift_report_page(train_df=train_df,test_df=test_df)
            return True
        except Exception as e:
            raise ApplicationException(e,sys) from e



    def initiate_data_validation(self):
        try:
            is_validated, validated_train_path, validated_test_path = self.is_Validation_successfull()
            
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.schema_path,
                is_validated=is_validated,
                message="Data_validation_performed ",
                validated_train_path=validated_train_path,
                validated_test_path=validated_test_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise ApplicationException(e, sys) from e


    def __del__(self):
        logging.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30}")
