
from Income.exception import ApplicationException
from Income.logger import logging
import os, sys
from Income.utils.utils import read_yaml
import pandas as pd
import collections
import pandas as pd
import numpy as np
from numpy import int64

class IngestedDataValidation:

    def __init__(self, validate_path, schema_path):
        try:
            self.validate_path = validate_path
            self.schema_path = schema_path
            self.data = read_yaml(self.schema_path)
        except Exception as e:
            logging.info(f"Error initializing IngestedDataValidation: {str(e)}")
            raise ApplicationException(e, sys) from e

    def validate_filename(self, file_name) -> bool:
        try:
            logging.info("Validating file name...")
            schema_file_name = self.data['File_Name']
            if schema_file_name == file_name:
                return True
        except Exception as e:
            logging.info(f"Error validating file name: {str(e)}")
            raise ApplicationException(e, sys) from e

    def missing_values_whole_column(self) -> bool:
        try:
            logging.info("Checking for missing values in whole column...")
            df = pd.read_csv(self.validate_path)
            count = 0
            for column in df:
                if (len(df[column]) - df[column].count()) == len(df[column]):
                    count += 1
            return True if count == 0 else False
        except Exception as e:
            logging.info(f"Error checking missing values in whole column: {str(e)}")
            raise ApplicationException(e, sys) from e
        
    def replace_null_values_with_nan(self):
        try:
            logging.info("Replacing null values with NaN...")
            
            # Log the shape before replacing null values
            df = pd.read_csv(self.validate_path)
            logging.info(f"Shape before replacing null values with NaN: {df.shape}")
            
            df.fillna(np.nan, inplace=True)
            
            # Log the shape after replacing null values
            logging.info(f"Shape after replacing null values with NaN: {df.shape}")
        except Exception as e:
            logging.info(f"Error replacing null values with NaN: {str(e)}")
            raise ApplicationException(e, sys) from e
    def check_column_names(self) -> bool:
        try:
            logging.info("Checking column names...")
            df = pd.read_csv(self.validate_path)
            df_column_names = df.columns
            schema_column_names = list(self.data['column_names'].keys())
            
           # logging.info(f"Data Column names {df_column_names}")
            
           # logging.info(f"Schema Column names {schema_column_names}")

            return True if collections.Counter(df_column_names) == collections.Counter(schema_column_names) else False
        except Exception as e:
            logging.info(f"Error checking column names: {str(e)}")
            raise ApplicationException(e, sys) from e
        
        
    def validate_data_types(self):
        flag = True  # Initialize the flag as True
        df=pd.read_csv(self.validate_path)
        expected_data_types = {
            'age': int64 ,
            'workclass': object,
            'fnlwgt': int64,
            'education': object,
            'educational-num': int64,
            'marital-status': object,
            'occupation': object,
            'relationship': object,
            'race': object,
            'gender': object,
            'capital-gain': int64,
            'capital-loss': int64,
            'hours-per-week': int64,
            'native-country': object,
            'income': object
        }

        for column, expected_type in expected_data_types.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in the dataset.")
            if not df[column].dtype == expected_type:
                flag = False  # Set flag to False if there is a data type mismatch
                raise TypeError(f"Data type mismatch for column '{column}'. Expected {expected_type}, but found {df[column].dtype}.")

        return flag  # Return the flag after validation