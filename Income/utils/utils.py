import sys 

import yaml
from Income.exception import ApplicationException




def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data
    except Exception as e:
        ApplicationException(e,sys)
