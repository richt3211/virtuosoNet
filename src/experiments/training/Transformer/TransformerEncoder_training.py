import os
import pickle
from src.constants import CACHE_MODEL_DIR
from src.models.model_run_job import ModelJob, ModelJobParams
from src.models.transformer import TransformerEncoder, TransformerEncoderHyperParams
from dataclasses import dataclass
from neptune.experiments import Experiment

import torch
import logging
import neptune

@dataclass
class TransformerEncoderJobParams(ModelJobParams):
    learning_rate:float = 3e-5
    grad_clip:float = 0.5
    model_name:str = "TRANSFORMER ENCODER ONLY"

class TransformerEncoderJob(ModelJob):
    def __init__(self, params:TransformerEncoderJobParams, model:TransformerEncoder, exp:Experiment):
        super().__init__(params, model, exp)
        self.params = params

        # self.num_key_augmentation = 1
        self.model = model.to(self.params.device)

    def zero_grad_optim(self):
        self.optimizer.zero_grad()

    def init_optimizer(self, model):
        self.optimizer = torch.optim.AdamW(model.parameters(), self.params.learning_rate)
        # self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate) 
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def step_optimizer(self, model, total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.grad_clip)
        self.optimizer.step()

    # def save_params(self, folder):
    #     folder = f'{CACHE_MODEL_DIR}/{folder}'

    #     # save model hyperparameters
    #     model_params_file_name = f'{folder}/params.pickle'
    #     if not os.path.exists(model_params_file_name):
    #         with open(model_params_file_name, 'wb') as file:
    #             pickle.dump(self.model.params, file)
            
    #         # save to neptune
    #         self.exp.log_artifact(model_params_file_name, 'params.pickle')

    