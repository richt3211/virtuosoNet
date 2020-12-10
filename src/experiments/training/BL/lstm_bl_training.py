from src.models.lstm_bl import LSTMBaseline
from src.models.model_run_job import ModelJob, ModelJobParams
from dataclasses import dataclass
from neptune.experiments import Experiment 

import torch
import logging

@dataclass
class LSTMBaselineTrainingJobParams(ModelJobParams):
    learning_rate:float = 0.1
    grad_clip:float = 0.5

class LSTMBaselineTraining(ModelJob):
    
    def __init__(self, params:LSTMBaselineTrainingJobParams, model:LSTMBaseline, exp:Experiment):
        super().__init__(params, model, exp)
        self.params = params
        self.model = model.to(self.params.device)

    def zero_grad_optim(self):
        self.optimizer.zero_grad()

    def init_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), self.params.learning_rate)
        # self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate) 
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def step_optimizer(self, model, total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.grad_clip)
        self.optimizer.step()