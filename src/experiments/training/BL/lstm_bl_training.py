from src.models.BL import LSTMBaseline
from src.models.model_run_job import ModelJob, ModelJobParams
from dataclasses import dataclass

import torch
import logging

@dataclass
class LSTMBlTrainingJobParams(ModelJobParams):
    learning_rate:float = 0.1
    grad_clip:float = 0.5

    def __post_init__(self):
        logging.info('LSTM Baseline Training Job Params')
        super().__post_init__()

class LSTMBlTraining(ModelJob):
    
    def __init__(self, params:LSTMBlTrainingJobParams, model:LSTMBaseline):
        super().__init__(params, model)
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.grad_clip)
        self.optimizer.step()