import os
import shutil
from neptune import Session 
from neptune.experiments import Experiment
from datetime import datetime
from src.constants import SRC_DIR 
from src.keys import NEPTUNE_TOKEN

import neptune 
import zipfile

from src.logger import init_logger

def log_neptune_timeline(log:str, exp:Experiment):
    exp.log_text('timeline', f'{datetime.now()} - {log}')

def init_experiment(exp_name:str, exp_description:str, tags: list, params: dict=None, upload_files:list=None, logger = None):
    '''Initalizes and creates a neptune experiment. '''  

    neptune.init('richt3211/thesis', api_token=NEPTUNE_TOKEN)

    exp:Experiment = neptune.create_experiment(
        name=exp_name,
        description=exp_description,
        params=params,
        tags=tags,
        logger=logger,
        upload_source_files=upload_files
    )
    return exp

def init_evaluation(experiment_id: str, artifacts:list) -> Experiment:
    '''Initalizes and creates a neptune experiment.'''  

    cache_dir = './artifacts'
    for path in [cache_dir, 'source', 'trill_source']:
        if os.path.exists(path):
            shutil.rmtree(path)


    os.mkdir(cache_dir)
    # load the trill model and params
    exp:Experiment = get_experiment_by_id('THESIS-75')
    exp.download_artifact('trill_params.pickle', cache_dir)
    exp.download_artifact('trill_best.pth', cache_dir)
    exp.download_sources()

    with zipfile.ZipFile('source.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
        
    os.rename('./source', './trill_source')

    init_logger()
    exp:Experiment = get_experiment_by_id(experiment_id)
    
    for art in artifacts:
        exp.download_artifact(art, f'{cache_dir}')

    exp.download_sources()

    with zipfile.ZipFile('source.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

    return exp 

def get_experiment_by_id(id:str):
    session = Session.with_default_backend(NEPTUNE_TOKEN)
    project = session.get_project('richt3211/thesis')

    exp:Experiment = project.get_experiments(id)[0]
    
    return exp