import sys 
import pandas as pd
import yaml
import numpy as np
import dill  
import os
import sys 
from Income.exception import ApplicationException
from Income.logger import logging



def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except Exception as e:
        ApplicationException(e,sys)

def save_array_to_directory(array: np.array, directory_path: str, file_name: str, extension: str = '.npy'):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Add the extension to the file name
        file_name_with_extension = file_name + extension

        # Generate the file path
        file_path = os.path.join(directory_path, file_name_with_extension)

        # Save the array to the file path
        np.save(file_path, array)
    except Exception as e:
        ApplicationException(e,sys)

    
def save_object(file_path:str,obj):
    try:
        logging.info(f" file path{file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise ApplicationException(e,sys) from e