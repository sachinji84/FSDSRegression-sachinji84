
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)


# Python Command to exeute it in GIT terminal is ;
# DELL@SachinKPC MINGW64 /d/Mytech/IDE_workspace/ineuron_git_repo/FSDSRegression-sachin-repofolder/FSDSRegression-sachin (main)
# $ python D:\\Mytech\\IDE_workspace\\ineuron_git_repo\\FSDSRegression-sachin-repofolder\\FSDSRegression-sachin\\src\\pipeline\\training_pipeline.py
# {'LinearRegression': 0.9302873217953068, 'Lasso': 0.9347652476618369, 'Ridge': 0.9321366737740069, 'Elasticnet': 0.8614823172075561}

# ====================================================================================

# Best Model Found , Model Name : Lasso , R2 Score : 0.9347652476618369

# ====================================================================================


# PYTHON command to execute on pwsh
# PS D:\Mytech\IDE_workspace\sachinji84git\iNeuronGit\FSDSRegression-sachinji84> & d:/Mytech/IDE_workspace/sachinji84git/iNeuronGit/FSDSRegression-sachinji84/.venv/Scripts/Activate.ps1
# (.venv) PS D:\Mytech\IDE_workspace\sachinji84git\iNeuronGit\FSDSRegression-sachinji84> python D:\Mytech\IDE_workspace\sachinji84git\iNeuronGit\FSDSRegression-sachinji84\src\pipeline\training_pipeline.py
# {'LinearRegression': 0.9302873217953068, 'Lasso': 0.9347652476618369, 'Ridge': 0.9321366737740069, 'Elasticnet': 0.8614823172075561}

# ====================================================================================

# Best Model Found , Model Name : Lasso , R2 Score : 0.9347652476618369

# ====================================================================================

# (.venv) PS D:\Mytech\IDE_workspace\sachinji84git\iNeuronGit\FSDSRegression-sachinji84> 