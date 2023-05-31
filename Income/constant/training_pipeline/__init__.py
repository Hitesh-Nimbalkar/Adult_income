# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline"

SCHEMA_CONFIG_KEY='schema_config'
SCHEMA_DIR_KEY ='schema_dir'
SCHEMA_FILE_NAME='schema_file'


TARGET_COLUMN_KEY='target_column'
NUMERICAL_COLUMN_KEY='numerical_columns'
CATEGORICAL_COLUMN_KEY='categorical_columns'
DROP_COLUMN_KEY='drop_columns'

from Income.constant.training_pipeline.data_ingestion import *
from Income.constant.training_pipeline.data_validation import *
from Income.constant.training_pipeline.data_transformation import *
from Income.constant.training_pipeline.model_trainer import *