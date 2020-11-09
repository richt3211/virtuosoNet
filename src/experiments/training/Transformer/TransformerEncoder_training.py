from src.models.model_run_job import ModelJob, ModelJobParams
from src.models.Transformer import TransformerEncoder, TransformerEncoderHyperParams
from dataclasses import dataclass
import torch
import logging

@dataclass
class TransformerEncoderJobParams(ModelJobParams):
    learning_rate:float = 3e-5
    grad_clip:float = 0.5
    model_name:str = "TRANSFORMER ENCODER ONLY"

    def __post_init__(self):
        logging.info(f'Transformer Encoder Job params')
        super().__post_init__()

class TransformerEncoderJob(ModelJob):
    def __init__(self, params:TransformerEncoderJobParams, model:TransformerEncoder):
        super().__init__(params, model)
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


    