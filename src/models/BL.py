import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable
import random
import numpy as np
import math
import logging

from torch.tensor import Tensor
from src.old import model_constants as cons
from src.old.nnModel import ContextAttention

DROP_OUT = 0.1

QPM_INDEX = 0
# VOICE_IDX = 11
PITCH_IDX = 12
TEMPO_IDX = PITCH_IDX + 13
DYNAMICS_IDX = TEMPO_IDX + 5
LEN_DYNAMICS_VEC = 4
QPM_PRIMO_IDX = 4
TEMPO_PRIMO_IDX = -2
NUM_VOICE_FEED_PARAM = 2


NUM_PRIME_PARAM = 11


class LSTMBaseline(nn.Module):
    # def __init__(self, network_parameters, device, step_by_step=False):
    #     super(HAN_Integrated, self).__init__()

    def __init__(self):
        super().__init__()

        self.input_size = 78
        self.output_size = 11

        self.lstm = nn.LSTM(self.input_size, self.output_size)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        return lstm_out