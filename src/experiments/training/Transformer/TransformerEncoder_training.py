from src.models.model_run_job import ModelJob, ModelJobParams
from src.models.Transformer import TransformerEncoder, TransformerEncoderHyperParams

import torch


class TransformerEncoderJob(ModelJob):
    def __init__(self, params:ModelJobParams, model:TransformerEncoder):
        super().__init__(params, model)

        self.learning_rate = 0.5
        self.num_key_augmentation = 1
        self.grad_clip = 0.5
        # hyper_params = TransformerEncoderHyperParams()
        # self.model = TransformerEncoder(hyper_params).to(self.params.device)
        self.model = model.to(self.params.device)
        self.model_name = 'TRANSFORMER ENCODER ONLY'

    def zero_grad_optim(self):
        self.optimizer.zero_grad()

    def init_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), 0.01) 
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def step_optimizer(self, model, total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self.optimizer.step()


    