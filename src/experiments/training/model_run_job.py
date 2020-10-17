from src.logger import init_logger

import torch 
import numpy as np
import logging

logger = logging.getLogger()
class ModelRun():

    def __init__(self, device):
        torch.cuda.set_device(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vel_param_idx = 1
        self.dev_param_idx = 2
        self.articul_param_idx = 3
        self.pedal_param_idx = 4

        self.feature_loss_init = {
            'tempo': [],
            'vel': [],
            'dev': [],
            'articul': [],
            'pedal': [],
            'trill': [],
            'kld': []
        }

    def train(self, model, data, num_epochs):
        pass

    def calculate_mean_loss(self, feature_loss, loss=None):
        feature_loss_mean = {}
        for key, value in feature_loss.items():
            feature_loss_mean[key] = np.mean(value)
        return feature_loss_mean, np.mean(loss)

    def print_loss(self, feature_loss, loss):
        logging.info(f'Total Loss: {np.mean(loss)}')
        loss_string = "\t"
        for key, value in feature_loss.items():
            loss_string += f'{key}: {np.mean(value):.4} '
        logging.info(loss_string)
        logging.info("")

    def set_up_logger(self, log_file_path):
        fhandler = logging.FileHandler(filename=log_file_path, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.DEBUG)