from collections import namedtuple


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])


DataIngestionConfig=namedtuple("DataIngestionConfig",[
    "raw_data_dir",
    "ingested_train_dir",
    "ingested_test_dir"
    ])

DataValidationConfig=namedtuple("DataValidationConfig",["schema_file_path",
                                                        "validated_train_path",
                                                        "validated_test_path"])
DataTransformationConfig = namedtuple("DataTransformationConfig",[  "transformed_input_train_path",
                                                                    "transformed_target_train_path",
                                                                    "transformed_input_test_path",
                                                                    "transformed_target_test_path",
                                                                  "preprocessed_object_file_path",
                                                                  "feature_engineering_object_file_path"])



ModelTrainerConfig = namedtuple("ModelTrainerConfig",["trained_model_file_path","saved_model_file_path"])