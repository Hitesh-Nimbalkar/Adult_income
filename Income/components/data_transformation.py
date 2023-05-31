import os,sys
import pandas as pd
import numpy as np
import re


from Income.logger import logging
from Income.exception import ApplicationException
from Income.entity.config_entity import DataTransformationConfig
from Income.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact

from Income.utils.utils import read_yaml,save_array_to_directory,save_object
from Income.constant import *
from Income.constant.training_pipeline import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_transformer

# Class in this file 
    # DataTransformation 
    # Feature Engineering
    
# Code_Flow 
    # Data_Transformation -----> method : Feature_Eng_Pipeline -----> Feature Engineering 




class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,numerical_columns,categorical_columns,target_columns,drop_columns):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        

                                ############### Accesssing Column Labels #########################
                                
                                
                 #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              #
                                
                                
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_columns = target_columns
        self.columns_to_drop = drop_columns

        
                                ########################################################################
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")


    # Feature Engineering Pipeline 
    
    
    
    
                                ######################### Data Modification ############################
        
    def drop_rows_with_nan(self, X: pd.DataFrame):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        
        # Drop rows with NaN values
        X = X.dropna()
        #X.to_csv("Nan_values_removed.csv", index=False)
        
        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        
        logging.info("Dropped NaN values.")
        
        return X
 
    
    def replace_with_others(self,X,threshold):
        
        try:
            
            
            categorical_cols_to_others =['native-country']

            for col in categorical_cols_to_others:
                unique_percentage = X[col].nunique() / X[col].count() * 100

                if unique_percentage < threshold:
                    X[col] = X[col].cat.add_categories('others')
                    X[col].fillna('others', inplace=True)
                    X[col].where(X[col].isin(X[col].value_counts().nlargest(1).index), 'others', inplace=True)

            return X
        
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def data_type_modification(self,X):
        
     
        # Categorical column from schema file
        object_cols = self.categorical_columns

        for col in object_cols:
            X[col] = X[col].astype('category')

        return X
        
    def remove_duplicate_rows_keep_last(self,X):
        
        logging.info(f"DataFrame shape before removing duplicates: {X.shape}")
        num_before = len(X)
        X.drop_duplicates(inplace = True)
        num_after = len(X)
        
        num_duplicates = num_before - num_after
        logging.info(f"Removed {num_duplicates} duplicate rows")
        logging.info(f"DataFrame shape after removing duplicates: {X.shape}")
        
        return X

    def Filtering_Special_Characters(self,X:pd.DataFrame):
        try:
            
            
        
        # Filter out rows with '?' values
            logging.info(f" Shape Before replacing special charaters :{X.shape}")
            X = X.replace('?', np.nan)
            
            
            logging.info(f" Shape after replacing special charaters :{X.shape}")
            logging.info("Removed special characters")
           # X.to_csv('special_characters.csv',index=False)
        
            return X
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    def drop_columns(self,X:pd.DataFrame):
        try:
            columns=X.columns
            
            logging.info(f"Columns before drop  {columns}")
            drop_column_labels=self.columns_to_drop
            
            logging.info(f" Dropping Columns {drop_column_labels} ")
            
            X=X.drop(columns=drop_column_labels,axis=1)
            
            return X
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    def run_data_modification(self,data):
        
        # Dropping irrelavant columns 
        X=self.drop_columns(data)
        
        # Filtering special character "?"
        X=self.Filtering_Special_Characters(X)
        
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)
        
        # Removing duplicated rows 
        X=self.remove_duplicate_rows_keep_last(X)
        
        # Modifying datatype Object ---> categorical
        X=self.data_type_modification(X)
        
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)
        
        # Passing threshold in percentage 
        X=self.replace_with_others(X,0.02)
        
        return X
    
    
    
    
                                          ######################################################    
    
    
    
    
    
                                            ######################### Outiers ############################
    
    
    
    
    
    def detect_outliers(self, data):
        outliers = {}
        
        numeric_cols=self.numerical_columns
        
        # Loop through numeric columns
        for col in numeric_cols:
            # Calculate the lower and upper quantiles
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            
            # Calculate the interquartile range (IQR)
            iqr = q3 - q1
            
            # Define the lower and upper bounds for outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Detect outliers
            col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            
            # Save outliers for the column
            outliers[col] = col_outliers
            
            # Logging
            logging.info(f"Detected {len(col_outliers)} outliers in column '{col}'.")
        
        return outliers

    def outliers_replaced(self, data, outliers):
        # Loop through columns with outliers
        for col, col_outliers in outliers.items():
            # Calculate the median for the column
            col_median = data[col].median()
            
            # Replace outliers with the median
            data.loc[data[col].isin(col_outliers), col] = col_median
            
            # Logging
            logging.info(f"Replaced {len(col_outliers)} outliers in column '{col}' with median value: {col_median}.")
        
        return data
    
    def outlier(self,X):
        
        outliers=self.detect_outliers(X)
        X=self.outliers_replaced(X,outliers)

        return X
    
    
    
    
    
    def data_wrangling(self,X:pd.DataFrame):
        try:

            
            # Data Modification 
            data_modified=self.run_data_modification(data=X)
            
            logging.info(" Data Modification Done")
            
            # Outlier Detection and Removal
            outliers_removed:pd.DataFrame = self.outlier(data_modified)
            
           # outliers_removed.to_csv("outliers_removed.csv",index=False)
            
            logging.info(" Outliers detection and removal Done ")
            

            
            return outliers_removed
    
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    
    
    
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified=self.data_wrangling(X)
            
            numerical_columns = self.numerical_columns
            categorical_columns=self.categorical_columns
            target_column=self.target_columns
            
            
            col = numerical_columns+categorical_columns+target_column
            
            
            print("\n")
            logging.info(f"New Column Order {col}")
            print("\n")
            
            
            data_modified:pd.DataFrame = data_modified[col]
            
            data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")
            
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shapde Modified Data : {data_modified.shape}")
         
            
            arr = data_modified.values
                
            return arr
        except Exception as e:
            raise ApplicationException(e,sys) from e



class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            
                                ############### Accesssing Column Labels #########################
                                
                                
                                #           Schema.yaml -----> DataTransfomation 
            
            # Schema File path 
            self.schema_file_path = self.data_validation_artifact.schema_file_path
            
            # Reading data in Schema 
            self.schema = read_yaml(file_path=self.schema_file_path)
            
            # Column data accessed from Schema.yaml
            self.target_column_name = self.schema[TARGET_COLUMN_KEY]
            self.numerical_columns = self.schema[NUMERICAL_COLUMN_KEY] 
            self.categorical_columns = self.schema[CATEGORICAL_COLUMN_KEY]
            self.drop_columns=self.schema[DROP_COLUMN_KEY]
            
                                ########################################################################
        except Exception as e:
            raise ApplicationException(e,sys) from e



    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(numerical_columns=self.numerical_columns,
                                                                            categorical_columns=self.categorical_columns,
                                                                            target_columns=self.target_column_name,
                                                                            drop_columns=self.drop_columns))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def get_data_transformer_object(self):
        try:
            logging.info('Creating Data Transformer Object')
            
            #numerical_columns = self.numerical_columns
            #categorical_columns = self.categorical_columns
            
            numerical_indices = [0, 1]  # Specify the numerical indices
            categorical_indices = [2, 3, 4, 5, 6, 7, 8, 9]  # Specify the categorical indices

            # Define transformer for numerical columns
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Define transformer for categorical columns
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])

            # Create column transformer with numerical and categorical transformers and assigned indices
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numerical_transformer, numerical_indices),
                    ('categorical', categorical_transformer, categorical_indices)
                ]
            )

            logging.info('Data transformations created successfully')
            return preprocessor
        except Exception as e:
            logging.error('An error occurred during data transformation')
            raise ApplicationException(e, sys) from e
        




    def initiate_data_transformation(self):
        try:
            
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path
            test_file_path = self.data_validation_artifact.validated_test_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logging.info(f" Tranin columns {train_df.columns}")
            
            # Schema.yaml ---> Extracting target column name
            target_column_name = self.target_column_name
            numerical_columns = self.numerical_columns
            categorical_columns = self.categorical_columns
                        
            # Log column information
            logging.info("Numerical columns: {}".format(numerical_columns))
            logging.info("Categorical columns: {}".format(categorical_columns))
            logging.info("Target Column: {}".format(target_column_name))
            
            
            col = numerical_columns + categorical_columns + target_column_name
            # All columns 
            logging.info("All columns: {}".format(col))
            
            
            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            feature_eng_train_arr = fe_obj.fit_transform(train_df)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            feature_eng_test_arr = fe_obj.transform(test_df)
            
            
            
            # Converting featured engineered array into dataframe
            logging.info(f"Converting featured engineered array into dataframe.")
            
            feature_eng_train_df = pd.DataFrame(feature_eng_train_arr,columns=col)
            #feature_eng_train_df.to_csv('feature_eng_train_df.csv',index=False)
            
            logging.info(f"Feature Engineering - Train Completed")
            
            feature_eng_test_df = pd.DataFrame(feature_eng_test_arr,columns=col)
            #feature_eng_test_df.to_csv('feature_eng_test_df.csv',index=False)
            
            #logging.info(f" Columns in feature enginering test {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training and testing dataframe.")
            
            
            
            # Train and Test Dataframe
            target_column_name=self.target_column_name

            target_feature_train_df = feature_eng_train_df[target_column_name]
            input_feature_train_df = feature_eng_train_df.drop(columns = target_column_name,axis = 1)
             
            target_feature_test_df = feature_eng_test_df[target_column_name]
            input_feature_test_df = feature_eng_test_df.drop(columns = target_column_name,axis = 1)
            
                                            ######## TARGET COLUMN ##########
            # Create an instance of LabelEncoder
            
            logging.info("Label encoding target column")
            label_encoder = LabelEncoder()

            # Apply label encoding to the target column
            target_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            target_test_encoded = label_encoder.transform(target_feature_test_df)

            # Reshape the target arrays to 1D shape
            target_train_encoded = target_train_encoded.ravel()
            target_test_encoded = target_test_encoded.ravel()

            # Log the encoded values
            logging.info("Encoded target_train_encoded:")
            logging.info(target_train_encoded)
            logging.info("Encoded target_test_encoded:")
            logging.info(target_test_encoded)

            
          
                                                #############################
                        
                                    ############ Input Fatures transformation########
            ## Preprocessing 
            logging.info("*" * 20 + " Applying preprocessing object on training dataframe and testing dataframe " + "*" * 20)
            preprocessing_obj = self.get_data_transformer_object()

            col = numerical_columns + categorical_columns

            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            
            # Log the shape of train_arr
            logging.info(f"Shape of train_arr: {train_arr.shape}")

            # Log the shape of test_arr
            logging.info(f"Shape of test_arr: {test_arr.shape}")

            logging.info("Transformation completed successfully")

                                ###############################################################
            
            # Saving the Transformed array ----> csv 
            ## Saving training data 
            logging.info("Saving transformed input and target Arrays")
                    
            transformed_input_train_path = self.data_transformation_config.transformed_input_train_path
            save_array_to_directory(directory_path=transformed_input_train_path, array=train_arr,file_name=DATA_TRANSFORMATION_INPUT_TRAIN)

            transformed_target_train_path = self.data_transformation_config.transformed_target_train_path
            save_array_to_directory(directory_path=transformed_target_train_path, array=target_train_encoded,file_name=DATA_TRANSFORMATION_TARGET_TRAIN)

            transformed_input_test_path = self.data_transformation_config.transformed_input_test_path
            save_array_to_directory(directory_path=transformed_input_test_path, array=test_arr,file_name=DATA_TRANSFORMATION_INPUT_TEST)

            transformed_target_test_path = self.data_transformation_config.transformed_target_test_path
            save_array_to_directory(directory_path=transformed_target_test_path, array=target_test_encoded,file_name=DATA_TRANSFORMATION_TARGET_TEST)


            
            logging.info(f"Saving transformed TRAIN input arrays  {transformed_input_train_path}")
            logging.info(f"Saving transformed TRAIN target arrays {transformed_target_train_path}")
            logging.info(f"Saving transformed TEST input arrays  {transformed_input_test_path}")
            logging.info(f"Saving transformed TEST target arrays {transformed_target_test_path}")
            
            
            logging.info("Saving Feature Engineering Object")
            
            ### Saving FFeature engineering and preprocessor object 
            
            # Artifact 
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            
            # Saving in the working directory 
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            # Saving in Artifact 
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            
             # Saving in the working directory 
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,os.path.basename(preprocessing_object_file_path)),
                        obj=preprocessing_obj)
            
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                    message="Data transformation successfull.",
                                                                    transformed_input_train_path=transformed_input_train_path,
                                                                    transformed_target_train_path=transformed_target_train_path,
                                                                    transformed_input_test_path=transformed_input_test_path,
                                                                    transformed_target_test_path=transformed_target_test_path,
                                                                    preprocessed_object_file_path = preprocessing_object_file_path,
                                                                    feature_engineering_object_file_path = feature_engineering_object_file_path)


            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

