from torch._C import ClassType
from src.logger import init_logger 
from src.constants import CACHE_DATA_DIR
from src.data.data_reader.read_featurized_cache import read_featurized
from src.models.model_run_job import ModelJob, ModelJobParams

import os 
import logging
import torch.nn as nn 

def init_training_job(dir_name: str, model_name: str, job_description: str = None) -> None:
  '''Creates the job run folder if it doesn't exist
  Creates the logging file if it doesn't exist. If the logging 
  file already exists it will wipe it to start with a new one'''  

  if not os.path.exists('./runs'):
    os.mkdir('./runs')

  run_dir = f'./runs/{dir_name}'
  if not os.path.exists(run_dir):
    os.mkdir(run_dir)

  init_logger(f'{run_dir}/training_run.log')

  logging.info(f'Starting training job for model {model_name}')
  if job_description:
    logging.info(job_description)

def get_dev_data():
  path = f'{CACHE_DATA_DIR}/train/training_data_development.pickle'
  dev_data = read_featurized(path)
  logging.info('Reading Dev Data')
  return dev_data

def get_full_data():
  path = f'{CACHE_DATA_DIR}/train/training_data.pickle'
  dev_data = read_featurized(path)
  logging.info('Reading Full Data')
  return dev_data

def start_training(
  data, 
  version: float,
  num_epochs:int, 
  model:nn.Module, 
  job_class, 
  job_params:ModelJobParams = None, 
):
  if job_params == None:
    job_params = ModelJobParams()

  training_job = job_class(job_params, model)
  training_job.run_job(data, num_epochs, version=version)