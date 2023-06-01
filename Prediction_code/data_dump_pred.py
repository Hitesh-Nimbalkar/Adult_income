import pandas as pd
import json
import logging

class data_dump_prediction:
    def __init__(self, DATABASE_NAME_PREDICTION, COLLECTION_NAME_PREDICTION):
        self.data_base = DATABASE_NAME_PREDICTION
        self.collection_name = COLLECTION_NAME_PREDICTION

    def data_dump(self, filepath):
        df = pd.read_csv(filepath)
        print(f"Rows and columns: {df.shape}")

        # Convert dataframe to json so that we can dump these records into MongoDB
        df.reset_index(drop=True, inplace=True)
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        json_record = list(json.loads(df.T.to_json()).values())
        print(json_record[0])

        print("Data Uploaded")

        # Check if the database exists
        if self.data_base in self.client.list_database_names():
            print(f"The database {self.data_base} already exists")
            # Check if the collection exists
            if self.collection_name in self.client[self.data_base].list_collection_names():
                print(f"The collection {self.collection_name} already exists")
                # Drop the existing collection
                self.client[self.data_base][self.collection_name].drop()
                print(f"The collection {self.collection_name} is dropped and will be replaced with new data")
            else:
                print(f"The collection {self.collection_name} does not exist and will be created")
        else:
            # Create the database and collection
            print(f"The database {self.data_base} does not exist and will be created")
            db = self.client[self.data_base]
            col = db[self.collection_name]
            print(f"The collection {self.collection_name} is created")

        # Insert converted json record into MongoDB
        self.client[self.data_base][self.collection_name].insert_many(json_record)

        logging.info("Prediction Data Updated to MongoDB")





@app.route('/data_dump', methods=['POST'])
def upload_prediction():
    
    # Get the file from the request
    prediction_file = 'batch_Prediction/prediction_csv/predictions.csv'
    
    Database_name_default=DATABASE_NAME
    Collection_name_default_prediction='prediction'

    # Get the user-provided database and collection names, or use default values
    DATABASE_NAME_PREDICTION = request.form.get('database_name_prediction', Database_name_default)
    COLLECTION_NAME_PREDICTION = request.form.get('collection_name_prediction', Collection_name_default_prediction)

    # Create an instance of data_dump_prediction
    data_dumper = data_dump_prediction(DATABASE_NAME_PREDICTION, COLLECTION_NAME_PREDICTION)

    # Call the data_dump method with the uploaded file
    data_dumper.data_dump(prediction_file)

    return 'Data uploaded successfully'