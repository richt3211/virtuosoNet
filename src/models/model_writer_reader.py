from neptune.experiments import Experiment
from src.constants import CACHE_MODEL_DIR
# from src.models.model_run_job import ModelJobParams

import os
import shutil
import torch
import torch.nn as nn
import pickle 


def save_checkpoint(state, is_best, folder, params, exp:Experiment):
    '''Saves the version of the model at each epoch, and updates the best version 
    of the model'''

    folder = f'./artifacts'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_name = ''
    if params.is_dev:
        file_name = f'model_dev.pth'
    else:
        file_name = f'model.pth'
    filepath = f'{folder}/{file_name}'
    torch.save(state, filepath)
    exp.log_artifact(filepath, file_name)
    if is_best:
        file_name = ''
        if params.is_dev:
            file_name = f'model_dev_best.pth'
        else:
            file_name = f'model_best.pth'

        best_filepath = f'{folder}/{file_name}'
        shutil.copyfile(filepath, best_filepath)
        exp.log_artifact(best_filepath, file_name)

def save_params(folder:str, params, exp:Experiment):
    folder = f'./artifacts'

    # save model hyperparameters
    model_params_file_name = f'{folder}/params.pickle'
    if not os.path.exists(model_params_file_name):
        with open(model_params_file_name, 'wb') as file:
            pickle.dump(params, file)
        
        # save to neptune
        exp.log_artifact(model_params_file_name, 'params.pickle')

def read_checkpoint(model:nn.Module, filepath, device):
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    return model.load_state_dict(checkpoint['state_dict'])

def read_params(filepath):
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        return x

