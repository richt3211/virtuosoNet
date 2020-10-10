import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable
import random
import numpy as np
import math

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
        self.output_size = 20 

        self.lstm = nn.LSTM(self.input_size, self.output_size)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        return lstm_out


class HANBaselineHyperParams():
    def __init__(self):
        beat_size = 64
        measure_size = 64
        voice_size = 64
        num_tempo_info = 3
        num_dynamic_info = 0

        self.input_size = 78
        self.hidden_size = 64
        self.num_layers = 2

        self.encoder_size = 32
        self.encoder_layer_num = 2
        self.encoded_vector_size = 16        
        self.encoder_input_size = (self.hidden_size + beat_size + measure_size + voice_size) * 2 + NUM_PRIME_PARAM

        self.num_attention_head = 8

        self.final_hidden_size = 64
        self.final_input = (self.hidden_size + voice_size + beat_size +
                                 measure_size) * 2 + self.encoder_size + \
                                num_tempo_info + num_dynamic_info

        self.step_by_step = True
        self.drop_out = 0.1

class HAN_Baseline(nn.Module):
    def __init__(self, hyper_params: HANBaselineHyperParams):
        super().__init__()
        self.input_size = hyper_params.input_size 
        self.hidden_size = hyper_params.hidden_size 
        self.num_layers = hyper_params.num_layers 

        self.encoder_input_size = hyper_params.encoder_input_size
        self.encoder_size = hyper_params.encoder_size 
        self.encoder_layer_num = hyper_params.encoder_layer_num
        self.encoded_vector_size = hyper_params.encoded_vector_size

        self.num_attention_head = hyper_params.num_attention_head

        self.final_hidden_size = hyper_params.final_hidden_size
        self.final_input = hyper_params.final_input 

        self.step_by_step = hyper_params.step_by_step
        self.drop_out = hyper_params.drop_out

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=self.drop_out)

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
        )

        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True)

        if self.encoder_size % self.num_attention_head == 0:
            self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        else:
            self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.encoder_size * 2)

        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )

        self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=False, rand_threshold=0.2, return_z=False):
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]

        num_notes = x.size(1)

        note_out = self.note_fc(x)
        note_out, _ = self.lstm(note_out)

        if type(initial_z) is not bool:
            if type(initial_z) is str and initial_z == 'zero':
                zero_mean = torch.zeros(self.encoded_vector_size)
                one_std = torch.ones(self.encoded_vector_size)
                perform_z = self.reparameterize(zero_mean, one_std).to(self.device)
            # if type(initial_z) is list:
            #     perform_z = self.reparameterize(torch.Tensor(initial_z), torch.Tensor(initial_z)).to(self.device)
            elif not initial_z.is_cuda:
                perform_z = torch.Tensor(initial_z).to(self.device).view(1,-1)
            else:
                perform_z = initial_z.view(1,-1)
            perform_mu = 0
            perform_var = 0
        else:
            perform_concat = torch.cat((note_out, y), 2)
            perform_contracted = self.performance_contractor(perform_concat)
            perform_note_encoded, _ = self.performance_note_encoder(perform_contracted)

            perform_measure = self.make_higher_node(perform_note_encoded, self.performance_measure_attention,
                                                    beat_numbers, measure_numbers, start_index, lower_is_note=True)
            perform_style_encoded, _ = self.performance_encoder(perform_measure)
            # perform_style_reduced = perform_style_reduced.view(-1,self.encoder_input_size)
            # perform_style_node = self.sum_with_attention(perform_style_reduced, self.perform_attention)
            # perform_style_vector = perform_style_encoded[:, -1, :]  # need check
            perform_style_vector = self.performance_final_attention(perform_style_encoded)
            perform_z, perform_mu, perform_var = \
                self.encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)

        if return_z:
            total_perform_z = [perform_z]
            for i in range(20):
                temp_z = self.reparameterize(perform_mu, perform_var)
                total_perform_z.append(temp_z)
            total_perform_z = torch.stack(total_perform_z)
            mean_perform_z = torch.mean(total_perform_z, 0, True)

            return mean_perform_z

        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)
        perform_z = perform_z.view(-1)


        final_hidden = self.init_hidden(1, 1, x.size(0), self.final_hidden_size)
        
        qpm_primo = x[:, 0, QPM_PRIMO_IDX]
        tempo_primo = x[0, 0, TEMPO_PRIMO_IDX:]

        prev_out = torch.zeros(self.output_size).to(self.device)
        prev_tempo = prev_out[QPM_INDEX:QPM_INDEX+1]
        prev_beat = -1
        prev_beat_end = 0
        out_total = torch.zeros(num_notes, self.output_size).to(self.device)
        prev_out_list = []
        has_ground_truth = y.size(1) > 1
        for i in range(num_notes):
            out_combined = torch.cat((note_out[0, i, :], prev_out, qpm_primo, tempo_primo, perform_z)).view(1, 1, -1)
            out, final_hidden = self.output_lstm(out_combined, final_hidden)

            out = out.view(-1)
            out = self.fc(out)

            prev_out_list.append(out)
            prev_out = out
            out_total[i, :] = out
        out_total = out_total.view(1, num_notes, -1)
        return out_total, perform_mu, perform_var, note_out
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:, lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.cat(higher_nodes, dim=1).view(1,-1,lower_hidden_size)

        return higher_nodes

    def encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = self.reparameterize(mu, var)
        return z, mu, var

    def init_hidden(self, num_layer, num_direction, batch_size, hidden_size):
        h0 = torch.zeros(num_layer * num_direction, batch_size, hidden_size).to(self.device)
        return (h0, h0)

