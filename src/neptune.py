from neptune import Session 
from neptune.experiments import Experiment
from datetime import datetime
from src.constants import SRC_DIR 
from src.keys import NEPTUNE_TOKEN

import neptune 
import zipfile

def log_neptune_timeline(log:str, exp:Experiment):
    exp.log_text('timeline', f'{datetime.now()} - {log}')

def init_experiment(exp_name:str, exp_description:str, params: dict, tags: list, model_src_path:str, logger = None):
    '''Initalizes and creates a neptune experiment. '''  

    neptune.init('richt3211/thesis', api_token=NEPTUNE_TOKEN)

    model_run_file_path = f"{SRC_DIR}/models/model_run_job.py"
    exp:Experiment = neptune.create_experiment(
        name=exp_name,
        description=exp_description,
        params=params,
        tags=tags,
        logger=logger,
        upload_source_files=[model_src_path, model_run_file_path]
    )
    return exp

def get_experiment_by_id(id:str):
    session = Session.with_default_backend(NEPTUNE_TOKEN)
    project = session.get_project('richt3211/thesis')

    exp:Experiment = project.get_experiments(id)[0]
    
    return exp