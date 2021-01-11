from dataclasses import asdict
from typing import Dict
from neptune.experiments import Experiment
from src.constants import CACHE_MODEL_DIR
# from src.models.model_run_job import ModelJobParams

import os
import shutil
import torch
import torch.nn as nn
import pickle

from src.models.model_run_job import ModelJob, ModelJobParams 


def initialize_artifact_folder(folder:str):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_checkpoint(state, is_best:bool, is_dev:bool, exp:Experiment, folder:str, model_name:str=None):
    '''Saves the version of the model at each epoch, and updates the best version 
    of the model'''

    file_name = f'{model_name}_model' if model_name else 'model'
    if is_dev:
        file_name = f'{file_name}_dev.pth'
    else:
        file_name = f'{file_name}.pth'
    filepath = f'{folder}/{file_name}'
    torch.save(state, filepath)
    exp.log_artifact(filepath, file_name)
    if is_best:
        file_name = f'{model_name}_model' if model_name else 'model' 
        if is_dev:
            file_name = f'{file_name}_dev_best.pth'
        else:
            file_name = f'{file_name}_best.pth'

        best_filepath = f'{folder}/{file_name}'
        shutil.copyfile(filepath, best_filepath)
        exp.log_artifact(best_filepath, file_name)

def save_params(folder:str, params, exp:Experiment, file_name:str):
    # save model hyperparameters
    model_params_file_name = f'{folder}/{file_name}'
    if not os.path.exists(model_params_file_name):
        with open(model_params_file_name, 'wb') as file:
            pickle.dump(params, file)
        
        # save to neptune
        exp.log_artifact(model_params_file_name, file_name)

def read_checkpoint(model:nn.Module, filepath, device):
    # if torch.cuda.is_available():
    #     map_location = lambda storage, loc: storage.cuda()
    # else:
    #     map_location = 'cpu'
    # checkpoint = torch.load(filepath, map_location=map_location)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict']) 
    model.to(device)
    return model

def read_params(filepath):
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        return x

def read_training_params(filepath, exp:Experiment) -> ModelJobParams:
    def set_value_in_params(jp:ModelJobParams, dp:dict, field_name:str, cast_fn):
        if field_name in dp and field_name in asdict(jp):
            jp.__setattr__(field_name, cast_fn(dp[field_name]))

    # if the experiment artifacts do not contain the training params, pull them down from neptune and manually reconstruct them
    if not os.path.exists(filepath):
        dict_params = exp.get_parameters()
        job_params = ModelJobParams()

        # go through each field one by one. Easier than trying to do it dynamically 
        set_value_in_params(job_params, dict_params, 'device_num', int)
        set_value_in_params(job_params, dict_params, 'qpm_index', int)
        set_value_in_params(job_params, dict_params, 'vel_param_idx', int)
        set_value_in_params(job_params, dict_params, 'dev_param_idx', int)
        set_value_in_params(job_params, dict_params, 'articul_param_idx', int)
        set_value_in_params(job_params, dict_params, 'pedal_param_idx', int)
        
        
        set_value_in_params(job_params, dict_params, 'num_key_augmentation', int)
        set_value_in_params(job_params, dict_params, 'batch_size', int)
        set_value_in_params(job_params, dict_params, 'epochs', int)
        
        set_value_in_params(job_params, dict_params, 'num_tempo_param', int)
        set_value_in_params(job_params, dict_params, 'num_prime_param', int)
        
        set_value_in_params(job_params, dict_params, 'criterion', str)
        set_value_in_params(job_params, dict_params, 'tempo_loss', bool)
        
        set_value_in_params(job_params, dict_params, 'time_steps', int)
        set_value_in_params(job_params, dict_params, 'is_dev', bool)
        
        if 'articul_mask' in dict_params and 'articul_mask' in asdict(job_params):
            job_params.articul_mask = str(dict_params['articul_mask'])
        else:
            job_params.articul_mask = 'pedal'
        
        if 'tempo_weight' not in dict_params:
            job_params.tempo_weight = 1
            job_params.vel_weight = 1
            job_params.dev_weight = 1
            job_params.articul_weight = 1
            job_params.pedal_weight = 7
        else:
            set_value_in_params(job_params, dict_params, 'tempo_weight', float)
            set_value_in_params(job_params, dict_params, 'vel_weight', float)
            set_value_in_params(job_params, dict_params, 'dev_weight', float)
            set_value_in_params(job_params, dict_params, 'articul_weight', float)
            set_value_in_params(job_params, dict_params, 'pedal_weight', float)
        
        return job_params
    else:
        return read_params(filepath)