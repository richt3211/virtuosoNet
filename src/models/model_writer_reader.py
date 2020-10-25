from src.constants import CACHE_MODEL_DIR

import os
import shutil
import torch
import torch.nn as nn

def save_checkpoint(state, is_best, folder, version, is_dev):
    folder = f'{CACHE_MODEL_DIR}/{folder}'
    if not os.path.exists(folder):
        os.mkdir(folder)
    if is_dev:
        filepath = f'{folder}/v{version}_dev.pth';
    else:
        filepath = f'{folder}/v{version}.pth'
    torch.save(state, filepath)
    if is_best:
        if is_dev:
            best_filepath = f'{folder}/v{version}_dev_best.pth'
        else:
            best_filepath = f'{folder}/v{version}_best.pth'
        shutil.copyfile(filepath, best_filepath)

def read_checkpoint(model:nn.Module, filepath, device):
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    return model.load_state_dict(checkpoint['state_dict'])

