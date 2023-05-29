import pandas as pd
import json
import pymongo
import urllib.parse
from Income.constant import *
import yaml
from Income.constant.data_base import *

env_file_path = os.path.join(ROOT_DIR, 'env.yaml')

# Load environment variables from env.yaml
with open(env_file_path) as file:
    env_vars = yaml.safe_load(file)
username = env_vars.get('USER_NAME')
password = env_vars.get('PASS_WORD')

escaped_username = urllib.parse.quote_plus(username)
escaped_password = urllib.parse.quote_plus(password)

# Use the escaped username and password in the MongoDB connection string
mongo_db_url = f"mongodb+srv://{username}:{password}@rentalbike.5fi8zs7.mongodb.net/"

client = pymongo.MongoClient(mongo_db_url)

DATA_FILE_PATH = "adult.csv"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    # Convert dataframe to json so that we can dump these records into MongoDB
    df.reset_index(drop=True, inplace=True)
    if "_id" in df.columns.to_list():
        df = df.drop(columns=["_id"], axis=1)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    print("Data Uploaded")
    print(f"{client}")

    # Check if the database exists
    if DATABASE_NAME in client.list_database_names():
        print(f"The database {DATABASE_NAME} already exists")
        client[DATABASE_NAME][COLLECTION_NAME].drop()
        print(f"The collection {COLLECTION_NAME} is dropped and will be replaced with new data")
    else:
        # Create the database and collection
        print(f"The database {DATABASE_NAME} does not exist and will be created")
        db = client[DATABASE_NAME]
        col = db[COLLECTION_NAME]
        print(f"The collection {COLLECTION_NAME} is created")

    # Insert converted JSON records into MongoDB
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

    print("Data Dump Completed")