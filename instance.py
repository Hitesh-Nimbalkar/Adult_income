
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
from Income.utils.utils import load_numpy_array_data,load_pickle_object


BATCH_PREDICTION = "batch_prediction"
INSTANCE_PREDICTION="Instance_prediction"
input_file_path="SCMS_Delivery_History_Dataset.csv"
feature_engineering_file_path ="Prediction_Files/feat_eng.pkl"
transformer_file_path ="Prediction_Files/preprocessed.pkl"
modmodel_file_pathel ="saved_models/model.pkl"






import pandas as pd
import joblib

# Load the preprocessor and machine learning model
preprocessor = load_pickle_object('Prediction_Files/preprocessed.pkl')
model = load_pickle_object('saved_models/model.pkl')


class instance_prediction_class:
    def __init__(self, age, hours_per_week, workclass, education, marital_status, occupation, relationship, race, gender, native_country) -> None:
        self.age = age
        self.hours_per_week = hours_per_week
        self.workclass = workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.gender = gender
        self.native_country = native_country

    def preprocess_input(self):
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'age': [self.age],
            'hours-per-week': [self.hours_per_week],
            'workclass': [self.workclass],
            'education': [self.education],
            'marital-status': [self.marital_status],
            'occupation': [self.occupation],
            'relationship': [self.relationship],
            'race': [self.race],
            'gender': [self.gender],
            'native-country': [self.native_country]
        })

        # Preprocess the user input using the preprocessor
        preprocessed_array = preprocessor.transform(user_input)

       
        

        # Return the preprocessed input as a DataFrame
        return preprocessed_array

    def predict_income(self, preprocessed_input):
        # Make a prediction using the pre-trained model
        predicted_price = model.predict(preprocessed_input)

        # Return the array of predicted prices
        return predicted_price

    def predict_price_from_input(self):
        # Preprocess the input using the preprocessor
        preprocessed_input = self.preprocess_input()

        # Make a prediction using the pre-trained model
        predicted_prices = self.predict_income(preprocessed_input)

        # Round off the predicted shipment prices to two decimal places
        rounded_prices = [round(price, 2) for price in predicted_prices]

        # Print the rounded predicted shipment prices
        for i, price in enumerate(rounded_prices):
            print(f"The predicted income  price for sample {i+1} is: $ {price}")

        return price