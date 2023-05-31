from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "train_file_path",
    "test_file_path", 
    "is_ingested", 
    "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",["schema_file_path",
                                                              "is_validated","message",
                                                              "validated_train_path",
                                                              "validated_test_path"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",["is_transformed",
                                                                    "message",
                                                                    "transformed_input_train_path",
                                                                    "transformed_target_train_path",
                                                                    "transformed_input_test_path",
                                                                    "transformed_target_test_path",
                                                                    "preprocessed_object_file_path",
                                                                    "feature_engineering_object_file_path"])


ModelTrainerArtifact = namedtuple("ModelTrainerArtifact",["is_trained",
                                                        "message",
                                                        "trained_model_object_file_path","saved_model_file_path",
                                                        "saved_model_report",
                                                        "model_artifact_report"])

ModelEvaluationArtifact=namedtuple("ModelEvaluationArtifact",["model_name","F1_Score","accuracy","model","model_report_path"])

ModelPusherArtifact=namedtuple("ModelPusherArtifact",["message"])