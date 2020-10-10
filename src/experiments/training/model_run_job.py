import torch 
import numpy as np

class ModelRun():

    def __init__(self, device):
        torch.cuda.set_device(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vel_param_idx = 1
        self.dev_param_idx = 2
        self.articul_param_idx = 3
        self.pedal_param_idx = 4

    def train(self, model, data, num_epochs):
        pass

    def print_loss(feature_loss, loss=None):
        if loss:
            print(f'\tTotal Loss: {np.mean(loss)}')
        print('\t', end="")
        for key, value in feature_loss.items():
            print(f'{key} loss: {np.mean(value):.4} ', end="")
        print()