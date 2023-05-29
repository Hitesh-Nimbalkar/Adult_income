import os
import pymongo
import yaml
from Income.constant import *
from Income.constant.data_base import DATABASE_NAME

import certifi
import urllib.parse

ca = certifi.where()

class MongoDBClient:
    client = None
    
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # Get the absolute path to the env.yaml file
                env_file_path = os.path.join(ROOT_DIR, 'env.yaml')
                
                # Load environment variables from env.yaml
                with open(env_file_path) as file:
                    env_vars = yaml.safe_load(file)
                username = env_vars.get('USER_NAME')
                password = env_vars.get('PASS_WORD')

                # Use the escaped username and password in the MongoDB connection string
                mongo_db_url = f"mongodb+srv://{username}:{password}@rentalbike.5fi8zs7.mongodb.net/"

                    
                print(mongo_db_url)
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise e