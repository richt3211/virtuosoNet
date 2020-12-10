import json
from typing import List, NewType
from torch._C import ClassType
from src.logger import init_logger
from src.constants import CACHE_DATA_DIR, ROOT_DIR, SRC_DIR
from src.data.data_reader.read_featurized_cache import read_featurized
from src.models.model_run_job import ModelJob, ModelJobParams
from src.keys import NEPTUNE_TOKEN

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import os 
import logging
import neptune
import shutil
from neptune.experiments import Experiment

from src.models.params import Params
from src.neptune import init_experiment, log_neptune_timeline

def init_training_job(is_dev:bool, exp_name:str, exp_description:str, hyper_params:Params, job_params: ModelJobParams, upload_files:list, tags:list = None) -> Experiment:
  '''Initalizes and creates a neptune experiment. '''  

  hyper_params_dict = asdict(hyper_params)
  job_params_dict = asdict(job_params)
  hyper_params_dict.update(job_params_dict)
  hyper_params_dict['device'] = f'cuda:{hyper_params.device_num}' 
  print(json.dumps(hyper_params_dict, indent=4))

  if os.path.exists('./artifacts'):
    shutil.rmtree('./artifacts')
    
  logger = init_logger()
  exp_tags = [f'{"dev" if is_dev else "full"}'] + tags
  exp:Experiment = init_experiment(
    exp_name=exp_name, 
    exp_description=exp_description, 
    tags=exp_tags,
    params=hyper_params_dict, 
    upload_files=upload_files,
    logger=logger
  )
  logger.info('Starting experiment')
  log_neptune_timeline('Starting experiment', exp)
  return exp

def init_legacy_training_job(is_dev:bool, exp_name:str, exp_description:str, params:dict, tags:list=None):
  '''Initalizes and creates a neptune experiment. '''  

  if os.path.exists('./artifacts'):
    shutil.rmtree('./artifacts')
    
  logger = init_logger()
  exp_tags = [f'{"dev" if is_dev else "full"}'] + tags
  upload_files = [f'{SRC_DIR}/old/nnModel.py', f'{SRC_DIR}/old/model_run.py']
  exp:Experiment = init_experiment(
    exp_name=exp_name, 
    exp_description=exp_description, 
    params=params, 
    tags=exp_tags, 
    upload_files=upload_files, 
    logger=logger
  )
  logger.info('Starting experiment')
  log_neptune_timeline('Starting experiment', exp)
  return exp

def get_dev_data(exp:Experiment):
  message = 'Reading Dev Data'
  log_neptune_timeline(message, exp)
  logging.info(message)
  path = f'{CACHE_DATA_DIR}/train/training_data_development.pickle'
  dev_data = read_featurized(path, exp)
  return dev_data

def get_full_data(exp:Experiment):
  message = 'Reading Full Data'
  log_neptune_timeline(message, exp)
  logging.info(message)
  path = f'{CACHE_DATA_DIR}/train/training_data.pickle'
  dev_data = read_featurized(path, exp)
  return dev_data

def run_training_experiment(
  exp_name:str, 
  exp_description:str, 
  tags:list, 
  is_dev:bool, 
  hyper_params:Params, 
  job_params:Params, 
  model_file_path:str, 
  model_folder:str, 
  model_class, 
  job_class
):
  model_run_path = f"{SRC_DIR}/models/model_run_job.py"
  upload_files = [model_file_path, model_run_path]
  exp = init_training_job(
    is_dev=is_dev,
    exp_name=exp_name,
    exp_description=exp_description,
    hyper_params=hyper_params,
    job_params=job_params,
    tags=tags,
    upload_files=upload_files
  )

  if is_dev:
    data = get_dev_data(exp)
  else:
    data = get_full_data(exp)

  model = model_class(hyper_params)
  training_job = job_class(job_params, model, exp)
  training_job.run_job(data, model_folder=model_folder)
  


def legacy_training_run_str(model_name:str, model_code:str, exp_description:str, exp_name:str, is_dev:str, tags:list):
  tags_arg = ','.join(tags)
  model_run_script_path = f'{ROOT_DIR}/virtuosoNet/src/old/model_run.py'
  data_path = f'{CACHE_DATA_DIR}/train/training_data_development' if is_dev == 'true' else f'{CACHE_DATA_DIR}/train/training_data'
  run_str = f'{model_run_script_path} -mode=train -code={model_code} -data={data_path} -model_name={model_name}  -is_dev={is_dev} -exp_name={exp_name} -exp_description={exp_description} -tags={tags_arg}'
  return run_str

# is_dev=True
# hyper_params = TransformerEncoderHyperParams()
# job_params = TransformerEncoderJobParams(is_dev=is_dev, epochs=50)
# model_file_path = f"{SRC_DIR}/models/transformer.py"
# run_training_experiment(
#     exp_name="With Tempo Loss",
#     exp_description="Running with tempo loss instead of note loss"
#     tags=['transformer_encoder']
#     is_dev=is_dev,
#     hyper_params=hyper_params,
#     job_params=job_params,
#     model_file_path=model_file_path,
#     model_folder="transfomer/transformer_encoder",
#     model_class=TransformerEncoder,
#     job_class=TransformerEncoderJob
# )
