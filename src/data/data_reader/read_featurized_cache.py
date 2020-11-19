import pickle
import logging 
from neptune.experiments import Experiment

def read_featurized(file_path, exp:Experiment):
    with open(file_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    train_xy = complete_xy['train']
    test_xy = complete_xy['valid']
    exp.log_text('timeline', f'number of train performances: {len(train_xy)} number of valid perf: {len(test_xy)}')
    exp.log_text('timeline', f'training sample example: {train_xy[0][0][0]}')

    return complete_xy

def read_featurized_stats(file_path):
    with open(file_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        means, stds = u.load()
        return means, stds