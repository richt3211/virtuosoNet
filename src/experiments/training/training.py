from typing import List
from torch._C import ClassType
from src.logger import init_logger 
from src.constants import CACHE_DATA_DIR
from src.data.data_reader.read_featurized_cache import read_featurized
from src.models.model_run_job import ModelJob, ModelJobParams

import matplotlib.pyplot as plt
import numpy as np
import os 
import logging

def init_training_job(is_dev:bool, dir_name: str, model_name: str, job_description: str = None) -> None:
  '''Creates the job run folder if it doesn't exist
  Creates the logging file if it doesn't exist. If the logging 
  file already exists it will wipe it to start with a new one'''  

  if not os.path.exists('./runs'):
    os.mkdir('./runs')

  run_dir = f'./runs/{dir_name}{"_dev" if is_dev else ""}'
  if not os.path.exists(run_dir):
    os.mkdir(run_dir)

  init_logger(f'{run_dir}/training_run.log')

  logging.info(f'Starting training job for model {model_name}')
  if job_description:
    logging.info(job_description)

def get_dev_data():
  logging.info('Reading Dev Data')
  path = f'{CACHE_DATA_DIR}/train/training_data_development.pickle'
  dev_data = read_featurized(path)
  return dev_data

def get_full_data():
  logging.info('Reading Full Data')
  path = f'{CACHE_DATA_DIR}/train/training_data.pickle'
  dev_data = read_featurized(path)
  return dev_data

def start_training(
  data, 
  version:float, 
  num_epochs:int, 
  job, 
  job_params:ModelJobParams, 
  model_class, 
  model_hyper_params, 
  model_folder:str
):
  model = model_class(model_hyper_params)
  training_job = job(job_params, model)
  return training_job.run_job(data, num_epochs, version=version, model_folder=model_folder)

def plot_loss(train_loss:List, valid_loss:List, folder_name:str, plot_title:str, is_dev:bool):
  max_loss = max(np.max(train_loss), np.max(valid_loss))
  X = [i for i in range(len(train_loss))]
  X = np.array(X)
  fig = plt.figure(figsize=(15,8))
  # fig = plt.figure()
  ax = fig.add_subplot(111)

  plt.plot(X, train_loss, color='b', marker='o', label='Training Loss')
  plt.plot(X, valid_loss, color='g', marker='o', label='Valid Loss')
  plt.legend(loc="upper right")
  # ax.legend([train_plt, valid_plt])
  # plt.legend([train_plt, valid_plt], ['Training Loss', 'Valid Loss'])
  plt.ylim((0, max_loss + max_loss * 0.1))
  plt.title(plot_title)
  plt.xlabel('epoch number')
  plt.ylabel('MSE')

  plot_path = f'./runs/{folder_name}{"_dev" if is_dev else ""}/loss_plot.png'
  plt.savefig(plot_path, bbox_inches='tight')
  plt.show()

def legacy_training_run_str(run_folder:str, model_folder:str, model_name:str, model_code:str, description:str, is_dev:str):
  model_run_script_path = '../../../../src/old/model_run.py'
  data_path = f'{CACHE_DATA_DIR}/train/training_data_development' if is_dev == 'true' else f'{CACHE_DATA_DIR}/train/training_data'
  run_str = f'{model_run_script_path} -mode=train -code={model_code} -data={data_path} -run_folder={run_folder} -model_name={model_name} -model_folder={model_folder} -is_dev={is_dev} -run_description={description}'
  return run_str
