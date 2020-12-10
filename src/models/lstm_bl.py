import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable
import random
import numpy as np
import math
import logging

from torch.tensor import Tensor
from src.models.params import Params
from src.old import model_constants as cons
from src.old.nnModel import ContextAttention
from dataclasses import dataclass

@dataclass
class LSTMBaselineHyperParams(Params):
    num_layers:int = 3
    hidden_size:int = 256
    dropout:float = 0.1

class LSTMBaseline(nn.Module):
    def __init__(self, params: LSTMBaselineHyperParams):
        super().__init__()
        self.params = params

        self.lstm_encoder = nn.LSTM(
            input_size=self.params.input_size, 
            hidden_size=self.params.hidden_size, 
            num_layers=self.params.num_layers,
            dropout=self.params.dropout,
            # bidirectional=True
        )
        self.decoder = nn.Linear(self.params.hidden_size, self.params.output_size)
        # self.init_weights()

    # def init_weights(self):
    #     y = self.params.input_size
    #     for weight in self.lstm_encoder.parameters():
    #         init.normal_(weight, 0.0, 1/np.sqrt(y))
    #     self.decoder.weight.data.normal_(0.0, (1/np.sqrt(y)))

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm_encoder(x)
        out = self.decoder(lstm_out)
        return out