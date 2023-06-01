from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from batch import batch_prediction
from instance import instance_prediction_class
import os
from Income.logger import logging
from Income.constant import *
from Income.entity.artifact_entity import ModelEvaluationArtifact

input_file_path = "adult.csv"
feature_engineering_file_path = "Prediction_Files/feat_eng.pkl"
transformer_file_path = "Prediction_Files/preprocessed.pkl"
model_file_path = "saved_models/model.pkl"

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'csv'}

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/batch", methods=["POST"])
def perform_batch_prediction():
    file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
    directory_path = UPLOAD_FOLDER
    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    
    # Check if the file has a valid extension
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        # Delete all files in the file path
        for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save the new file to the uploads directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(file_path)
        
        logging.info("CSV received and Uploaded")

        # Perform batch prediction using the uploaded file
        batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
        batch.start_batch_prediction()

        output = "Batch Prediction Done"
        return render_template("index.html", prediction_result=output, prediction_type='batch')
    else:
        return render_template('index.html', prediction_type='batch', error='Invalid file type')

@app.route("/instance", methods=["POST"])
def perform_instance_prediction():
    age = int(request.form['age'])
    hours_per_week = int(request.form['hours_per_week'])
    workclass = request.form['workclass']
    education = request.form['education']
    marital_status = request.form['marital_status']
    occupation = request.form['occupation']
    relationship = request.form['relationship']
    race = request.form['race']
    gender = request.form['gender']
    native_country = request.form['native_country']

    predictor = instance_prediction_class(age, hours_per_week, workclass, education, marital_status, occupation, relationship, race, gender, native_country)
    predicted_income = predictor.predict_price_from_input()
    if predicted_income == 0:
        predicted_income_text = "<=50K"
    elif predicted_income == 1:
        predicted_income_text = ">=50K"
    else:
        predicted_income_text = "Unknown"

    return render_template('index.html', prediction_type='instance', predicted_income_text=predicted_income_text)



if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8000  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)
    
    
    