import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable
import random
import numpy
import math
from src.old import model_constants as cons

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

class LSTM_Baseline(nn.Module):
    # def __init__(self, network_parameters, device, step_by_step=False):
    #     super(HAN_Integrated, self).__init__()

    def __init__(self):
        super().__init__()

        self.input_size = 78
        self.output_size = 20 

        self.lstm = nn.LSTM(self.input_size, self.output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        return lstm_out


# spits out a trained model using a very simple LSTM
def train_model(data, model):
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        for xy_tuple in data['train']:
            x = xy_tuple[0]
            y = xy_tuple[1]
            # align_matched = xy_tuple[3]
            # pedal_status = xy_tuple[4]
            # edges = xy_tuple[5]

            input = torch.tensor(x, dtype=torch.float)
            target = torch.tensor(y, dtype=torch.float)
            pred = model(input)

            loss = loss_function(pred, target)
            print(f'Loss is: {loss}')
            loss.backward()
            optimizer.step()

    # output the validation loss
    with torch.no_grad():
        for xy_tuple in data['valid']:
            x = xy_tuple[0]
            y = xy_tuple[1]
            # align_matched = xy_tuple[3]
            # pedal_status = xy_tuple[4]
            # edges = xy_tuple[5]

            input = torch.tensor(x, dtype=torch.float)
            target = torch.tensor(y, dtype=torch.float)
            pred = model(input)

            loss = loss_function(pred, target)
            print(f'Validation Loss is: {loss}')

    return model