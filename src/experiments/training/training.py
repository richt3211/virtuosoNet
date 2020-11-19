import json
from typing import List, NewType
from torch._C import ClassType
from src.logger import init_logger 
from src.constants import CACHE_DATA_DIR
from src.data.data_reader.read_featurized_cache import read_featurized
from src.models.model_run_job import ModelJob, ModelJobParams
from src.keys import NEPTUNE_TOKEN
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import os 
import logging
import neptune
from neptune.experiments import Experiment

from src.models.params import Params

def init_training_job(is_dev:bool, exp_name:str, exp_description:str, hyper_params:Params, job_params: ModelJobParams, tags:list = None) -> Experiment:
  '''Initalizes and creates a neptune experiment. '''  

  neptune.init('richt3211/thesis', api_token=NEPTUNE_TOKEN)

  hyper_params_dict = asdict(hyper_params)
  job_params_dict = asdict(job_params)
  hyper_params_dict.update(job_params_dict)
  print(json.dumps(hyper_params_dict, indent=4))

  exp_tags = [f'{"dev" if is_dev else "full"}'] + tags
  exp:Experiment = neptune.create_experiment(
    name=exp_name,
    description=exp_description,
    params=hyper_params_dict,
    tags=exp_tags,
  )
  exp.log_text('timeline', 'Starting experiment')
  return exp

def get_dev_data(exp:Experiment):
  exp.log_text('timeline', 'Reading Dev Data')
  path = f'{CACHE_DATA_DIR}/train/training_data_development.pickle'
  dev_data = read_featurized(path, exp)
  return dev_data

def get_full_data(exp:Experiment):
  exp.log_text('timeline','Reading Full Data')
  path = f'{CACHE_DATA_DIR}/train/training_data.pickle'
  dev_data = read_featurized(path, exp)
  return dev_data

def start_training(
  data, 
  version:float, 
  num_epochs:int, 
  job, 
  job_params:ModelJobParams, 
  model_class, 
  model_hyper_params, 
  model_folder:str,
  exp: Experiment
):
  model = model_class(model_hyper_params)
  training_job = job(job_params, model, exp)
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
