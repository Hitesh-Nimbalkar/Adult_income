import sys
from Income.constant import *
from Income.logger import logging
from Income.exception import ApplicationException
from Income.entity.config_entity import *
from Income.utils.utils import read_yaml 
from Income.constant.training_pipeline import *


class Configuration:

    def __init__(self,config_file_path:str =CONFIG_FILE_PATH,current_time_stamp:str = CURRENT_TIME_STAMP
        ) -> None:
        try:
            self.config_info  = read_yaml(file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def get_data_ingestion_config(self) ->DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir=os.path.join(
                artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR,
                self.time_stamp
            )
            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
                                        )

            ingested_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
            )
            ingested_train_dir = os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]
            )
            ingested_test_dir =os.path.join(
                ingested_data_dir,
                data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]
            )
            

            data_ingestion_config=DataIngestionConfig(
                raw_data_dir=raw_data_dir, 
                ingested_train_dir=ingested_train_dir, 
                ingested_test_dir=ingested_test_dir
            )
            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_artifact_dir = os.path.join(artifact_dir, 
                                                            DATA_TRANSFORMATION_ARTIFACT_DIR, 
                                                            self.time_stamp)

            data_transformation_config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            preprocessed_object_file_path = os.path.join(data_transformation_artifact_dir,
                                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY])

            feature_engineering_object_file_path = os.path.join(data_transformation_artifact_dir,
                                data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                data_transformation_config[DATA_TRANSFORMATION_FEATURE_ENGINEERING_FILE_NAME_KEY])

            transformed_input_train_path = os.path.join(data_transformation_artifact_dir,
                                data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                                data_transformation_config[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
            
            transformed_target_train_path=os.path.join(data_transformation_artifact_dir,
                                data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                                data_transformation_config[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])

            transformed_input_test_path = os.path.join(data_transformation_artifact_dir,
                                data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                                data_transformation_config[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])
            
            transformed_target_test_path=os.path.join(data_transformation_artifact_dir,
                                            data_transformation_config[DATA_TRANSFORMATION_DIR_NAME_KEY],
                                            data_transformation_config[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])

            data_transformation_config = DataTransformationConfig(transformed_input_train_path=transformed_input_train_path,
                                                                  transformed_target_train_path=transformed_target_train_path,
                                                    transformed_input_test_path=transformed_input_test_path,
                                                    transformed_target_test_path=transformed_target_test_path,
                                                    preprocessed_object_file_path=preprocessed_object_file_path,
                                                    feature_engineering_object_file_path=feature_engineering_object_file_path)
            
            
            ### Folder Structure ####
            # Artifact 
                # data_Transformation 
                    # Transformed Train 
                
                    
                    # Transformed Test 
                      
                        
                    # Preprocessor 

            
            
            logging.info(f"Data Transformation Config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
        
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_trainer_artifact_dir = os.path.join(artifact_dir, 
                                                      MODEL_TRAINER_ARTIFACT_DIR, 
                                                      self.time_stamp)

            model_trainer_config = self.config_info[MODEL_TRAINER_CONFIG_KEY]

            trained_model_file_path = os.path.join(model_trainer_artifact_dir,
                                                   model_trainer_config[MODEL_TRAINER_TRAINED_MODEL_DIR],
                                                   model_trainer_config[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY])
            saved_model_file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,MODEL_FILE_NAME)
 
            model_trainer_config = ModelTrainerConfig(trained_model_file_path=trained_model_file_path,
                                                      saved_model_file_path=saved_model_file_path)
            logging.info(f"Model Trainer Config : {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
        
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            data_validation_config=self.config_info[DATA_VALIDATION_CONFIG_KEY]
            
            schema_dir=self.config_info[SCHEMA_CONFIG_KEY]

            data_validation_artifact_dir=os.path.join(
                                                        artifact_dir,
                                                        data_validation_config[DATA_VALIDATION_ARTIFACT_DIRECTORY_KEY],
                                                        self.time_stamp
                                                         )
            
            validated_path=os.path.join(data_validation_artifact_dir,data_validation_config[DATA_VLIDATION_DIRECTORY_KEY])
            
            validated_train_path=os.path.join(data_validation_artifact_dir,validated_path,data_validation_config[DATA_VALIDATION_TRAIN_DIR_KEY])
            
            validated_test_path=os.path.join(data_validation_artifact_dir,validated_path,data_validation_config[DATA_VALIDATION_TEST_DIR_KEY])


            schema_file_path = os.path.join(
                ROOT_DIR,
                schema_dir[SCHEMA_DIR_KEY],
                schema_dir[SCHEMA_FILE_NAME]
            )
            
            print(schema_file_path)
            

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                validated_train_path=validated_train_path,
                validated_test_path=validated_test_path
            )
            return data_validation_config
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    def get_training_pipeline_config(self) ->TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )

            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training pipleine config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise ApplicationException(e,sys) from e