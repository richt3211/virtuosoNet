import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, network_parameters, device, step_by_step=False):
        super(LSTM_Baseline, self).__init__()
        self.device = device
        self.step_by_step = step_by_step
        self.is_graph = network_parameters.is_graph
        self.is_teacher_force = network_parameters.is_teacher_force
        self.is_baseline = network_parameters.is_baseline
        self.num_graph_iteration = network_parameters.graph_iteration
        self.hierarchy = network_parameters.hierarchy_level
        if hasattr(network_parameters, 'is_test_version') and network_parameters.is_test_version:
            self.test_version = True
        else:
            self.test_version = False
        # self.is_simplified_note = network_parameters.is_simplified

        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.num_layers = network_parameters.note.layer
        self.hidden_size = network_parameters.note.size
        self.num_beat_layers = network_parameters.beat.layer
        self.beat_hidden_size = network_parameters.beat.size
        self.num_measure_layers = network_parameters.measure.layer
        self.measure_hidden_size = network_parameters.measure.size

        self.final_hidden_size = network_parameters.final.size
        self.num_voice_layers = network_parameters.voice.layer
        self.voice_hidden_size = network_parameters.voice.size
        self.final_input = network_parameters.final.input
        if self.test_version:
            self.final_input -= 1
        self.encoder_size = network_parameters.encoder.size
        self.encoded_vector_size = network_parameters.encoded_vector_size
        self.encoder_input_size = network_parameters.encoder.input
        self.encoder_layer_num = network_parameters.encoder.layer
        self.num_attention_head = network_parameters.num_attention_head
        self.num_edge_types = network_parameters.num_edge_types

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
        )

        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.final_hidden_size, self.output_size)

        self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True)
        
        # if self.encoder_size % self.num_attention_head == 0:
            # self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        # else:
            # self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.encoder_size * 2)
        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
        # self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
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
        # perform_z = self.performance_decoder(perform_z)
        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)
        perform_z = perform_z.view(-1)

        final_hidden = self.init_hidden(1, 1, x.size(0), self.final_hidden_size)
        if self.step_by_step:
            qpm_primo = x[:, 0, QPM_PRIMO_IDX]
            tempo_primo = x[0, 0, TEMPO_PRIMO_IDX:]

            if self.is_teacher_force:
                true_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)

            # prev_out = y[0, 0, :]
            # prev_tempo = y[:, 0, QPM_INDEX]
            prev_out = torch.zeros(self.output_size).to(self.device)
            prev_tempo = prev_out[QPM_INDEX:QPM_INDEX+1]
            prev_beat = -1
            prev_beat_end = 0
            out_total = torch.zeros(num_notes, self.output_size).to(self.device)
            prev_out_list = []
            # if args.beatTempo:
            #     prev_out[0] = tempos_spanned[0, 0, 0]
            has_ground_truth = y.size(1) > 1
            if self.is_baseline:
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
            else:
                for i in range(num_notes):
                    current_beat = beat_numbers[start_index + i] - beat_numbers[start_index]
                    current_measure = measure_numbers[start_index + i] - measure_numbers[start_index]
                    if current_beat > prev_beat:  # beat changed
                        if i - prev_beat_end > 0:  # if there are outputs to consider
                            corresp_result = torch.stack(prev_out_list).unsqueeze_(0)
                        else:  # there is no previous output
                            corresp_result = torch.zeros((1,1,self.output_size-1)).to(self.device)
                        result_node = self.result_for_tempo_attention(corresp_result)
                        prev_out_list = []
                        result_nodes[current_beat, :] = result_node

                        if self.is_teacher_force and current_beat > 0 and random.random() < rand_threshold:
                            prev_tempo = true_tempos[0,current_beat-1,:]

                        tempos = torch.zeros(1, num_beats, 1).to(self.device)
                        if self.test_version:
                            beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :],
                                                        measure_hidden_out[0, current_measure, :], prev_tempo, x[0,i,self.input_size-2:self.input_size-1],
                                                        result_nodes[current_beat, :],
                                                        measure_perform_style[0, current_measure, :])).view(1, 1, -1)
                        else:
                            beat_tempo_vec = x[0, i, TEMPO_IDX:TEMPO_IDX + 5]
                            beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :], measure_hidden_out[0, current_measure,:], prev_tempo,
                                                        qpm_primo, tempo_primo, beat_tempo_vec,
                                                        result_nodes[current_beat, :], perform_z)).view(1, 1, -1)
                        beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)

                        tmp_tempos = self.beat_tempo_fc(beat_forward)

                        prev_beat_end = i
                        prev_tempo = tmp_tempos.view(1)
                        prev_beat = current_beat

                    tmp_voice = voice_numbers[start_index + i] - 1

                    if self.is_teacher_force and i > 0 and random.random() < rand_threshold:
                        prev_out = torch.cat((prev_tempo, y[0, i - 1, 1:]))

                    if self.test_version:
                        out_combined = torch.cat(
                            (note_out[0, i, :], beat_hidden_out[0, current_beat, :],
                                measure_hidden_out[0, current_measure, :],
                                prev_out, x[0,i,self.input_size-2:], measure_perform_style[0, current_measure,:])).view(1, 1, -1)
                    else:
                        out_combined = torch.cat(
                            (note_out[0, i, :], beat_hidden_out[0, current_beat, :],
                                measure_hidden_out[0, current_measure, :],
                                prev_out, qpm_primo, tempo_primo, perform_z)).view(1, 1, -1)
                    out, final_hidden = self.output_lstm(out_combined, final_hidden)
                    # out = torch.cat((out, out_combined), 2)
                    out = out.view(-1)
                    out = self.fc(out)

                    prev_out_list.append(out)
                    out = torch.cat((prev_tempo, out))

                    prev_out = out
                    out_total[i, :] = out

                out_total = out_total.view(1, num_notes, -1)
                hidden_total = torch.cat((note_out, beat_out_spanned, measure_out_spanned), 2)
                return out_total, perform_mu, perform_var, hidden_total
            else:  # non autoregressive
                qpm_primo = x[:,:,QPM_PRIMO_IDX].view(1,-1,1)
                tempo_primo = x[:,:,TEMPO_PRIMO_IDX:].view(1,-1,2)
                # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
                beat_qpm_primo = qpm_primo[0,0,0].repeat((1, num_beats, 1))
                beat_tempo_primo = tempo_primo[0,0,:].repeat((1, num_beats, 1))
                beat_tempo_vector = self.note_tempo_infos_to_beat(x, beat_numbers, start_index, TEMPO_IDX)
                if 'beat_hidden_out' not in locals():
                    beat_hidden_out = beat_out_spanned
                num_beats = beat_hidden_out.size(1)
                # score_z_beat_spanned = score_z.repeat(num_beats,1).view(1,num_beats,-1)
                perform_z_beat_spanned = perform_z.repeat(num_beats,1).view(1,num_beats,-1)
                beat_tempo_cat = torch.cat((beat_hidden_out, beat_qpm_primo, beat_tempo_primo, beat_tempo_vector, perform_z_beat_spanned), 2)
                beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)
                tempos = self.beat_tempo_fc(beat_forward)
                num_notes = note_out.size(1)
                tempos_spanned = self.span_beat_to_note_num(tempos, beat_numbers, num_notes, start_index)
                # y[0, :, 0] = tempos_spanned.view(-1)

                # mean_velocity_info = x[:, :, mean_vel_start_index:mean_vel_start_index+4].view(1,-1,4)
                # dynamic_info = torch.cat((x[:, :, mean_vel_start_index + 4].view(1,-1,1),
                #                           x[:, :, vel_vec_start_index:vel_vec_start_index + 4]), 2).view(1,-1,5)

                out_combined = torch.cat((
                    note_out, beat_out_spanned, measure_out_spanned,
                    # qpm_primo, tempo_primo, mean_velocity_info, dynamic_info,
                    perform_z_batched), 2)

                out, final_hidden = self.output_lstm(out_combined, final_hidden)

                out = self.fc(out)
                # out = torch.cat((out, trill_out), 2)

                out = torch.cat((tempos_spanned, out), 2)
                score_combined = torch.cat((
                    note_out, beat_out_spanned, measure_out_spanned), 2)

                return out, perform_mu, perform_var, score_combined