import torch
import pickle
import argparse
import math
import numpy as np
import shutil
import os
import matplotlib
import pyScoreParser.xml_matching as xml_matching
import pyScoreParser.performanceWorm as perf_worm
import src.old.data_process as dp
import copy
import random
from src.old import nnModel
import src.old.model_parameters as param
import src.old.model_constants as cons
from datetime import datetime

import sys

import os 
print(os.getcwd())
print(os.path.abspath('../../../data'))


sys.modules['xml_matching'] = xml_matching

matplotlib.use('Agg')

# DATA_DIR = 'data'

class ModelRun():

    def __init__(self, args, data_dir):
        random.seed(0)

        self.DATA_DIR = data_dir
        # self.args = parser.parse_args()
        self.args = args
        self.LOSS_TYPE = self.args.trainingLoss
        self.HIERARCHY = False
        self.IN_HIER = False
        self.HIER_MEAS = False
        self.HIER_BEAT = False
        self.HIER_MODEL = None
        self.RAND_TRAIN = self.args.randomTrain
        self.TRILL = False

        if 'measure' in self.args.modelCode or 'beat' in self.args.modelCode:
            self.HIERARCHY = True
        elif 'note' in self.args.modelCode:
            self.IN_HIER = True  # In hierarchy mode
        if self.HIERARCHY or self.IN_HIER:
            if 'measure' in self.args.modelCode or 'measure' in self.args.hierCode:
                self.HIER_MEAS = True
            elif 'beat' in self.args.modelCode or 'beat' in self.args.hierCode:
                self.HIER_BEAT = True

        if 'trill' in self.args.modelCode:
            self.TRILL = True

        ### parameters
        self.learning_rate = 0.0003
        self.TIME_STEPS = 500
        self.VALID_STEPS = 5000
        self.DELTA_WEIGHT = 2
        self.NUM_UPDATED = 0
        weight_decay = 1e-5
        self.GRAD_CLIP = 5
        self.KLD_MAX = 0.01
        self.KLD_SIG = 20e4
        if self.args.sessMode == 'train':
            print(
                'Learning Rate: {}, Time_steps: {}, Delta weight: {}, Weight decay: {}, Grad clip: {}, KLD max: {}, KLD sig step: {}'.format
                (self.learning_rate, self.TIME_STEPS, self.DELTA_WEIGHT, weight_decay, self.GRAD_CLIP, self.KLD_MAX,
                 self.KLD_SIG))

        self.num_epochs = 100
        self.num_key_augmentation = 1

        self.NUM_INPUT = 78
        self.NUM_PRIME_PARAM = 11
        if self.HIERARCHY:
            self.NUM_OUTPUT = 2
        elif self.TRILL:
            self.NUM_INPUT += self.NUM_PRIME_PARAM
            self.NUM_OUTPUT = 5
        else:
            self.NUM_OUTPUT = 11
        if self.IN_HIER:
            self.NUM_INPUT += 2

        self.NUM_TEMPO_PARAM = 1
        self.VEL_PARAM_IDX = 1
        self.DEV_PARAM_IDX = 2
        self.PEDAL_PARAM_IDX = 3
        num_second_param = 0
        self.num_trill_param = 5
        num_voice_feed_param = 0  # velocity, onset deviation
        num_tempo_info = 0
        num_dynamic_info = 0  # distance from marking, dynamics vector 4, mean_piano, forte marking and velocity = 4
        is_trill_index_score = -11
        self.is_trill_index_concated = -11 - (self.NUM_PRIME_PARAM + num_second_param)

        with open(f'{self.DATA_DIR}/train/{self.args.dataName}_stat.dat', "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            if self.args.trainingLoss == 'CE':
                self.MEANS, self.STDS, self.BINS = u.load()
                new_prime_param = 0
                new_trill_param = 0
                for i in range(self.NUM_PRIME_PARAM):
                    new_prime_param += len(self.BINS[i]) - 1
                for i in range(self.NUM_PRIME_PARAM, self.NUM_PRIME_PARAM + self.num_trill_param - 1):
                    new_trill_param += len(self.BINS[i]) - 1
                self.NUM_PRIME_PARAM = new_prime_param
                print('New self.NUM_PRIME_PARAM: ', self.NUM_PRIME_PARAM)
                self.num_trill_param = new_trill_param + 1
                self.NUM_OUTPUT = self.NUM_PRIME_PARAM + self.num_trill_param
                self.NUM_TEMPO_PARAM = len(self.BINS[0]) - 1
            else:
                self.MEANS, self.STDS = u.load()

        self.QPM_INDEX = 0
        # VOICE_IDX = 11
        # TEMPO_IDX = 26
        self.QPM_PRIMO_IDX = 4
        TEMPO_PRIMO_IDX = -2
        self.GRAPH_KEYS = ['onset', 'forward', 'melisma', 'rest']
        if self.args.slurEdge:
            self.GRAPH_KEYS.append('slur')
        if self.args.voiceEdge:
            self.GRAPH_KEYS.append('voice')
        self.N_EDGE_TYPE = len(self.GRAPH_KEYS) * 2
        # mean_vel_start_index = 7
        # vel_vec_start_index = 33

        self.batch_size = 1

        torch.cuda.set_device(self.args.device)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.args.sessMode == 'train' and not self.args.resumeTraining:
            NET_PARAM = param.initialize_model_parameters_by_code(self.args.modelCode)
            NET_PARAM.num_edge_types = self.N_EDGE_TYPE
            NET_PARAM.training_args = self.args
            param.save_parameters(NET_PARAM, self.args.modelCode + '_param')
        elif self.args.resumeTraining:
            NET_PARAM = param.load_parameters(self.args.modelCode + '_param')
        else:
            NET_PARAM = param.load_parameters(self.args.modelCode + '_param')
            TrillNET_Param = param.load_parameters(self.args.trillCode + '_param')
            # if not hasattr(NET_PARAM, 'num_edge_types'):
            #     NET_PARAM.num_edge_types = 10
            # if not hasattr(TrillNET_Param, 'num_edge_types'):
            #     TrillNET_Param.num_edge_types = 10
            self.TRILL_MODEL = nnModel.TrillRNN(TrillNET_Param, self.DEVICE).to(self.DEVICE)

        if 'isgn' in self.args.modelCode:
            self.MODEL = nnModel.ISGN(NET_PARAM, self.DEVICE).to(self.DEVICE)
        elif 'han' in self.args.modelCode:
            if 'ar' in self.args.modelCode:
                step_by_step = True
            else:
                step_by_step = False
            self.MODEL = nnModel.HAN_Integrated(NET_PARAM, self.DEVICE, step_by_step).to(self.DEVICE)
        elif 'trill' in self.args.modelCode:
            self.MODEL = nnModel.TrillRNN(NET_PARAM, self.DEVICE).to(self.DEVICE)
        else:
            print('Error: Unclassified model code')
            # Model = nnModel.HAN_VAE(NET_PARAM, self.DEVICE, False).to(self.DEVICE)

        self.optimizer = torch.optim.Adam(self.MODEL.parameters(), lr=self.learning_rate, weight_decay=weight_decay)

    def criterion(self,pred, target, aligned_status=1):
        if self.LOSS_TYPE == 'MSE':
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = torch.sum(aligned_status).item() * pred.shape[-1]
                if data_size == 0:
                    data_size = 1
            if target.shape != pred.shape:
                print('Error: The shape of the target and prediction for the loss calculation is different')
                print(target.shape, pred.shape)
                return torch.zeros(1).to(self.DEVICE)
            return torch.sum(((target - pred) ** 2) * aligned_status) / data_size
        elif self.LOSS_TYPE == 'CE':
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = torch.sum(aligned_status).item() * pred.shape[-1]
                if data_size == 0:
                    data_size = 1
                    print('data size for loss calculation is zero')
            return -1 * torch.sum(
                (target * torch.log(pred) + (1 - target) * torch.log(1 - pred)) * aligned_status) / data_size

    def save_checkpoint(self, state, is_best, filename='', model_name='prime'):
        if filename == '':
            filename = self.args.modelCode
        save_name = model_name + '_' + filename + '_checkpoint.pth.tar'
        torch.save(state, save_name)
        if is_best:
            best_name = model_name + '_' + filename + '_best.pth.tar'
            shutil.copyfile(save_name, best_name)

    def edges_to_matrix(self, edges, num_notes):
        if not self.MODEL.is_graph:
            return None
        num_keywords = len(self.GRAPH_KEYS)
        matrix = np.zeros((self.N_EDGE_TYPE, num_notes, num_notes))

        for edg in edges:
            if edg[2] not in self.GRAPH_KEYS:
                continue
            edge_type = self.GRAPH_KEYS.index(edg[2])
            matrix[edge_type, edg[0], edg[1]] = 1
            if edge_type != 0:
                matrix[edge_type + num_keywords, edg[1], edg[0]] = 1
            else:
                matrix[edge_type, edg[1], edg[0]] = 1

        matrix[num_keywords, :, :] = np.identity(num_notes)
        matrix = torch.Tensor(matrix)
        return matrix

    def edges_to_matrix_short(self, edges, slice_index):
        if not self.MODEL.is_graph:
            return None
        num_keywords = len(self.GRAPH_KEYS)
        num_notes = self.slice_idx[1] - self.slice_idx[0]
        matrix = np.zeros((self.N_EDGE_TYPE, num_notes, num_notes))
        start_edge_index = xml_matching.binary_index_for_edge(edges, slice_index[0])
        end_edge_index = xml_matching.binary_index_for_edge(edges, slice_index[1] + 1)
        for i in range(start_edge_index, end_edge_index):
            edg = edges[i]
            if edg[2] not in self.GRAPH_KEYS:
                continue
            if edg[1] >= slice_index[1]:
                continue
            edge_type = self.GRAPH_KEYS.index(edg[2])
            matrix[edge_type, edg[0] - slice_index[0], edg[1] - slice_index[0]] = 1
            if edge_type != 0:
                matrix[edge_type + num_keywords, edg[1] - slice_index[0], edg[0] - slice_index[0]] = 1
            else:
                matrix[edge_type, edg[1] - slice_index[0], edg[0] - slice_index[0]] = 1
        matrix[num_keywords, :, :] = np.identity(num_notes)
        matrix = torch.Tensor(matrix)

        return matrix

    def edges_to_sparse_tensor(self, edges):
        num_keywords = len(self.GRAPH_KEYS)
        edge_list = []
        edge_type_list = []

        for edg in edges:
            edge_type = self.GRAPH_KEYS.index(edg[2])
            edge_list.append(edg[0:2])
            edge_list.append([edg[1], edg[0]])
            edge_type_list.append(edge_type)
            if edge_type != 0:
                edge_type_list.append(edge_type + num_keywords)
            else:
                edge_type_list.append(edge_type)

            edge_list = torch.LongTensor(edge_list)
        edge_type_list = torch.FloatTensor(edge_type_list)

        matrix = torch.sparse.FloatTensor(edge_list.t(), edge_type_list)

        return matrix

    def categorize_value_to_vector(self, y, bins):
        vec_length = sum([len(x) for x in bins])
        num_notes = len(y)
        y_categorized = []
        num_categorized_params = len(bins)
        for i in range(num_notes):
            note = y[i]
            total_vec = []
            for j in range(num_categorized_params):
                temp_vec = [0] * (len(bins[j]) - 1)
                temp_vec[int(note[j])] = 1
                total_vec += temp_vec
            total_vec.append(note[-1])  # add up trill
            y_categorized.append(total_vec)

        return y_categorized

    def scale_model_prediction_to_original(self, prediction, MEANS, STDS):
        for i in range(len(STDS)):
            for j in range(len(STDS[i])):
                if STDS[i][j] < 1e-4:
                    STDS[i][j] = 1
        prediction = np.squeeze(np.asarray(prediction.cpu()))
        num_notes = len(prediction)
        if self.LOSS_TYPE == 'MSE':
            for i in range(11):
                prediction[:, i] *= STDS[1][i]
                prediction[:, i] += MEANS[1][i]
            for i in range(11, 15):
                prediction[:, i] *= STDS[1][i + 4]
                prediction[:, i] += MEANS[1][i + 4]
        elif self.LOSS_TYPE == 'CE':
            prediction_in_value = np.zeros((num_notes, 16))
            for i in range(num_notes):
                bin_range_start = 0
                for j in range(15):
                    feature_bin_size = len(self.BINS[j]) - 1
                    feature_class = np.argmax(prediction[i, bin_range_start:bin_range_start + feature_bin_size])
                    feature_value = (self.BINS[j][feature_class] + self.BINS[j][feature_class + 1]) / 2
                    prediction_in_value[i, j] = feature_value
                    bin_range_start += feature_bin_size
                prediction_in_value[i, 15] = prediction[i, -1]
            prediction = prediction_in_value

        return prediction

    def load_file_and_generate_performance(self, path_name, composer=None, z=None,
                                           start_tempo=None, return_features=False,
                                           multi_instruments=None):
        composer = self.args.composer if composer is None else composer
        z = self.args.latent if z is None else z
        start_tempo = self.args.startTempo if start_tempo is None else start_tempo
        multi_instruments = self.args.multi_instruments if multi_instruments is None else multi_instruments
        vel_pair = (int(self.args.velocity.split(',')[0]), int(self.args.velocity.split(',')[1]))
        test_x, xml_notes, xml_doc, edges, note_locations = xml_matching.read_xml_to_array(path_name, self.MEANS, self.STDS,
                                                                                           start_tempo, composer,
                                                                                           vel_pair)
        batch_x = torch.Tensor(test_x)
        num_notes = len(test_x)
        input_y = torch.zeros(1, num_notes, self.NUM_OUTPUT).to(self.DEVICE)

        if type(z) is dict:
            initial_z = z['z']
            qpm_change = z['qpm']
            z = z['key']
            batch_x[:, self.QPM_PRIMO_IDX] = batch_x[:, self.QPM_PRIMO_IDX] + qpm_change
        else:
            initial_z = 'zero'

        if self.IN_HIER:
            batch_x = batch_x.to(self.DEVICE).view(1, -1, self.HIER_MODEL.input_size)
            graph = self.edges_to_matrix(edges, batch_x.shape[1])
            self.MODEL.is_teacher_force = False
            if type(initial_z) is list:
                hier_z = initial_z[0]
                final_z = initial_z[1]
            else:
                # hier_z = [z] * self.HIER_MODEL_PARAM.encoder.size
                hier_z = 'zero'
                final_z = initial_z
            hier_input_y = torch.zeros(1, num_notes, self.HIER_MODEL.output_size)
            hier_output, _ = self.run_model_in_steps(batch_x, hier_input_y, graph, note_locations, initial_z=hier_z,
                                                model=self.HIER_MODEL)
            if 'measure' in self.args.hierCode:
                hierarchy_numbers = [x.measure for x in note_locations]
            else:
                hierarchy_numbers = [x.section for x in note_locations]
            hier_output_spanned = self.HIER_MODEL.span_beat_to_note_num(hier_output, hierarchy_numbers, len(test_x), 0)
            combined_x = torch.cat((batch_x, hier_output_spanned), 2)
            prediction, _ = self.run_model_in_steps(combined_x, input_y, graph, note_locations, initial_z=final_z,
                                               model=self.MODEL)
        else:
            if type(initial_z) is list:
                initial_z = initial_z[0]
            batch_x = batch_x.to(self.DEVICE).view(1, -1, self.NUM_INPUT)
            graph = self.edges_to_matrix(edges, batch_x.shape[1])
            prediction, _ = self.run_model_in_steps(batch_x, input_y, graph, note_locations, initial_z=initial_z,
                                               model=self.MODEL)

        trill_batch_x = torch.cat((batch_x, prediction), 2)
        trill_prediction, _ = self.run_model_in_steps(trill_batch_x, torch.zeros(1, num_notes, cons.num_trill_param), graph,
                                                 note_locations, model=self.TRILL_MODEL)

        prediction = torch.cat((prediction, trill_prediction), 2)
        prediction = self.scale_model_prediction_to_original(prediction, self.MEANS, self.STDS)

        output_features = xml_matching.model_prediction_to_feature(prediction)
        output_features = xml_matching.add_note_location_to_features(output_features, note_locations)
        if return_features:
            return output_features

        output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                               predicted=True)
        output_midi, midi_pedals = xml_matching.xml_notes_to_midi(output_xml, multi_instruments)
        piece_name = path_name.split('/')
        save_name = f'{self.DATA_DIR}/test_result/{piece_name[-2]}_by_{self.args.modelCode}_z{str(z)}'
        # save_name = 'test_result/' + piece_name[-2] + '_by_' + self.args.modelCode + '_z' + str(z)

        perf_worm.plot_performance_worm(output_features, save_name + '.png')
        print(f'Saving midi performance to {save_name}')
        xml_matching.save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_name + '.mid',
                                                   bool_pedal=self.args.boolPedal, disklavier=self.args.disklavier)

    def load_file_and_encode_style(self, path, perf_name, composer_name):
        test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(path, perf_name, composer_name,
                                                                                     self.MEANS, self.STDS)
        qpm_primo = test_x[0][4]

        test_x, test_y = self.handle_data_in_tensor(test_x, test_y, hierarchy_test=self.IN_HIER)
        edges = self.edges_to_matrix(edges, test_x.shape[0])

        if self.IN_HIER:
            test_x = test_x.view((1, -1, self.HIER_MODEL.input_size))
            hier_y = test_y[0].view(1, -1, self.HIER_MODEL.output_size)
            perform_z_high = self.encode_performance_style_vector(test_x, hier_y, edges, note_locations,
                                                             model=self.HIER_MODEL)
            hier_outputs, _ = self.run_model_in_steps(test_x, hier_y, edges, note_locations, model=self.HIER_MODEL)
            if self.HIER_MEAS:
                hierarchy_numbers = [x.measure for x in note_locations]
            elif self.HIER_BEAT:
                hierarchy_numbers = [x.beat for x in note_locations]
            hier_outputs_spanned = self.HIER_MODEL.span_beat_to_note_num(hier_outputs, hierarchy_numbers,
                                                                         test_x.shape[1], 0)
            input_concat = torch.cat((test_x, hier_outputs_spanned), 2)
            batch_y = test_y[1].view(1, -1, self.MODEL.output_size)
            perform_z_note = self.encode_performance_style_vector(input_concat, batch_y, edges, note_locations,
                                                             model=self.MODEL)
            perform_z = [perform_z_high, perform_z_note]

        else:
            batch_x = test_x.view((1, -1, self.NUM_INPUT))
            batch_y = test_y.view((1, -1, self.NUM_OUTPUT))
            perform_z = self.encode_performance_style_vector(batch_x, batch_y, edges, note_locations)
            perform_z = [perform_z]

        return perform_z, qpm_primo

    def encode_performance_style_vector(self, input, input_y, edges, note_locations, model=None):
        model = self.MODEL if model is None else model
        with torch.no_grad():
            model_eval = model.eval()
            if edges is not None:
                edges = edges.to(self.DEVICE)
            encoded_z = model_eval(input, input_y, edges,
                                   note_locations=note_locations, start_index=0, return_z=True)
        return encoded_z

    def encode_all_emotionNet_data(self, path_list, style_keywords):
        perform_z_by_emotion = []
        perform_z_list_by_subject = []
        qpm_list_by_subject = []
        num_style = len(style_keywords)
        if self.IN_HIER:
            num_model = 2
        else:
            num_model = 1
        for pair in path_list:
            subject_num = pair[2]
            for sub_idx in range(subject_num):
                indiv_perform_z = []
                indiv_qpm = []
                path = cons.emotion_folder_path + pair[0] + '/'
                composer_name = pair[1]
                for key in style_keywords:
                    perf_name = key + '_sub' + str(sub_idx + 1)
                    perform_z_li, qpm_primo = self.load_file_and_encode_style(path, perf_name, composer_name)
                    indiv_perform_z.append(perform_z_li)
                    indiv_qpm.append(qpm_primo)
                for i in range(1, num_style):
                    for j in range(num_model):
                        indiv_perform_z[i][j] = indiv_perform_z[i][j] - indiv_perform_z[0][j]
                    indiv_qpm[i] = indiv_qpm[i] - indiv_qpm[0]
                perform_z_list_by_subject.append(indiv_perform_z)
                qpm_list_by_subject.append(indiv_qpm)
        for i in range(num_style):
            z_by_models = []
            for j in range(num_model):
                emotion_mean_z = []
                for z_list in perform_z_list_by_subject:
                    emotion_mean_z.append(z_list[i][j])
                mean_perform_z = torch.mean(torch.stack(emotion_mean_z), 0, True)
                z_by_models.append(mean_perform_z)
            if i is not 0:
                emotion_qpm = []
                for qpm_change in qpm_list_by_subject:
                    emotion_qpm.append(qpm_change[i])
                mean_qpm_change = np.mean(emotion_qpm)
            else:
                mean_qpm_change = 0
            print(style_keywords[i], z_by_models, mean_qpm_change)
            perform_z_by_emotion.append({'z': z_by_models, 'key': style_keywords[i], 'qpm': mean_qpm_change})

        return perform_z_by_emotion
        # with open(self.args.testPath + self.args.perfName + '_style' + '.dat', 'wb') as f:
        #     pickle.dump(mean_perform_z, f, protocol=2)

    def run_model_in_steps(self, input, input_y, edges, note_locations, initial_z=False, model=None):
        model = self.MODEL if model is None else model
        num_notes = input.shape[1]
        with torch.no_grad():  # no need to track history in validation
            model_eval = model.eval()
            total_output = []
            total_z = []
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=self.VALID_STEPS,
                                                               overlap=False)
            # if edges is not None:
            #     edges = edges.to(self.DEVICE)

            for self.slice_idx in slice_indexes:
                batch_start, batch_end = self.slice_idx
                if edges is not None:
                    batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end].to(self.DEVICE)
                else:
                    batch_graph = None

                batch_input = input[:, batch_start:batch_end, :].view(1, -1, model.input_size)
                batch_input_y = input_y[:, batch_start:batch_end, :].view(1, -1, model.output_size)
                temp_outputs, perf_mu, perf_var, _ = model_eval(batch_input, batch_input_y, batch_graph,
                                                                note_locations=note_locations, start_index=batch_start,
                                                                initial_z=initial_z)
                total_z.append((perf_mu, perf_var))
                total_output.append(temp_outputs)

            outputs = torch.cat(total_output, 1)
            return outputs, total_z

    def batch_time_step_run(self, data, model, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        batch_start, batch_end = self.training_data['self.slice_idx']
        batch_x, batch_y = self.handle_data_in_tensor(data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

        batch_x = batch_x.view((batch_size, -1, self.NUM_INPUT))
        batch_y = batch_y.view((batch_size, -1, self.NUM_OUTPUT))

        align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end]).view((batch_size, -1, 1)).to(self.DEVICE)
        pedal_status = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view((batch_size, -1, 1)).to(self.DEVICE)

        if self.training_data['graphs'] is not None:
            edges = self.training_data['graphs']
            if edges.shape[1] == batch_end - batch_start:
                edges = edges.to(self.DEVICE)
            else:
                edges = edges[:, batch_start:batch_end, batch_start:batch_end].to(self.DEVICE)
        else:
            edges = self.training_data['graphs']

        prime_batch_x = batch_x
        if self.HIERARCHY:
            prime_batch_y = batch_y
        else:
            prime_batch_y = batch_y[:, :, 0:self.NUM_PRIME_PARAM]

        model_train = model.train()
        outputs, perform_mu, perform_var, total_out_list \
            = model_train(prime_batch_x, prime_batch_y, edges, self.note_locations, batch_start)

        if self.HIERARCHY:
            if self.HIER_MEAS:
                hierarchy_numbers = [x.measure for x in self.note_locations]
            elif self.HIER_BEAT:
                hierarchy_numbers = [x.beat for x in self.note_locations]
            tempo_in_hierarchy = self.MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, batch_start, 0)
            dynamics_in_hierarchy = self.MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, batch_start, 1)
            tempo_loss = self.criterion(outputs[:, :, 0:1], tempo_in_hierarchy)
            vel_loss = self.criterion(outputs[:, :, 1:2], dynamics_in_hierarchy)
            if self.args.deltaLoss and outputs.shape[1] > 1:
                vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
                vel_true_delta = dynamics_in_hierarchy[:, 1:, :] - dynamics_in_hierarchy[:, :-1, :]

                vel_loss += self.criterion(vel_out_delta, vel_true_delta) * self.DELTA_WEIGHT
                vel_loss /= 1 + self.DELTA_WEIGHT

            total_loss = tempo_loss + vel_loss
        elif self.TRILL:
            trill_bool = batch_x[:, :, self.is_trill_index_concated:self.is_trill_index_concated + 1]
            if torch.sum(trill_bool) > 0:
                total_loss = self.criterion(outputs, batch_y, trill_bool)
            else:
                return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(
                    1), torch.zeros(1)

        else:
            if 'isgn' in self.args.modelCode and self.args.intermediateLoss:
                total_loss = torch.zeros(1).to(self.DEVICE)
                for out in total_out_list:
                    if model.is_baseline:
                        tempo_loss = self.criterion(out[:, :, 0:1],
                                               prime_batch_y[:, :, 0:1], align_matched)
                    else:
                        tempo_loss = self.cal_tempo_loss_in_beat(out, prime_batch_y, self.note_locations, batch_start)
                    vel_loss = self.criterion(out[:, :, self.VEL_PARAM_IDX:self.DEV_PARAM_IDX],
                                         prime_batch_y[:, :, self.VEL_PARAM_IDX:self.DEV_PARAM_IDX], align_matched)
                    dev_loss = self.criterion(out[:, :, self.DEV_PARAM_IDX:self.PEDAL_PARAM_IDX],
                                         prime_batch_y[:, :, self.DEV_PARAM_IDX:self.PEDAL_PARAM_IDX], align_matched)
                    articul_loss = self.criterion(out[:, :, self.PEDAL_PARAM_IDX:self.PEDAL_PARAM_IDX + 1],
                                             prime_batch_y[:, :, self.PEDAL_PARAM_IDX:self.PEDAL_PARAM_IDX + 1], pedal_status)
                    pedal_loss = self.criterion(out[:, :, self.PEDAL_PARAM_IDX + 1:], prime_batch_y[:, :, self.PEDAL_PARAM_IDX + 1:],
                                           align_matched)

                    total_loss += (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11
                total_loss /= len(total_out_list)
            else:
                if model.is_baseline:
                    tempo_loss = self.criterion(outputs[:, :, 0:1],
                                           prime_batch_y[:, :, 0:1], align_matched)
                else:
                    tempo_loss = self.cal_tempo_loss_in_beat(outputs, prime_batch_y, self.note_locations, batch_start)
                vel_loss = self.criterion(outputs[:, :, self.VEL_PARAM_IDX:self.DEV_PARAM_IDX],
                                     prime_batch_y[:, :, self.VEL_PARAM_IDX:self.DEV_PARAM_IDX], align_matched)
                dev_loss = self.criterion(outputs[:, :, self.DEV_PARAM_IDX:self.PEDAL_PARAM_IDX],
                                     prime_batch_y[:, :, self.DEV_PARAM_IDX:self.PEDAL_PARAM_IDX], align_matched)
                articul_loss = self.criterion(outputs[:, :, self.PEDAL_PARAM_IDX:self.PEDAL_PARAM_IDX + 1],
                                         prime_batch_y[:, :, self.PEDAL_PARAM_IDX:self.PEDAL_PARAM_IDX + 1], pedal_status)
                pedal_loss = self.criterion(outputs[:, :, self.PEDAL_PARAM_IDX + 1:], prime_batch_y[:, :, self.PEDAL_PARAM_IDX + 1:],
                                       align_matched)
                total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11

        if isinstance(perform_mu, bool):
            perform_kld = torch.zeros(1)
        else:
            perform_kld = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
            total_loss += perform_kld * self.kld_weight
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.GRAD_CLIP)
        self.optimizer.step()

        if self.HIERARCHY:
            return tempo_loss, vel_loss, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), perform_kld
        elif self.TRILL:
            return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(
                1), total_loss, torch.zeros(1)
        else:
            return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), perform_kld

        # loss = self.criterion(outputs, batch_y)
        # tempo_loss = self.criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])

    def cal_tempo_loss_in_beat(self, pred_x, true_x, note_locations, start_index):
        previous_beat = -1

        num_notes = pred_x.shape[1]
        start_beat = note_locations[start_index].beat
        num_beats = note_locations[num_notes + start_index - 1].beat - start_beat + 1

        pred_beat_tempo = torch.zeros([num_beats, self.NUM_TEMPO_PARAM]).to(self.DEVICE)
        true_beat_tempo = torch.zeros([num_beats, self.NUM_TEMPO_PARAM]).to(self.DEVICE)
        for i in range(num_notes):
            current_beat = note_locations[i + start_index].beat
            if current_beat > previous_beat:
                previous_beat = current_beat
                if 'baseline' in self.args.modelCode:
                    for j in range(i, num_notes):
                        if note_locations[j + start_index].beat > current_beat:
                            break
                    if not i == j:
                        pred_beat_tempo[current_beat - start_beat] = torch.mean(pred_x[0, i:j, self.QPM_INDEX])
                        true_beat_tempo[current_beat - start_beat] = torch.mean(true_x[0, i:j, self.QPM_INDEX])
                else:
                    pred_beat_tempo[current_beat - start_beat] = pred_x[0, i, self.QPM_INDEX:self.QPM_INDEX + self.NUM_TEMPO_PARAM]
                    true_beat_tempo[current_beat - start_beat] = true_x[0, i, self.QPM_INDEX:self.QPM_INDEX + self.NUM_TEMPO_PARAM]

        tempo_loss = self.criterion(pred_beat_tempo, true_beat_tempo)
        if self.args.deltaLoss and pred_beat_tempo.shape[0] > 1:
            prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
            true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
            delta_loss = self.criterion(prediction_delta, true_delta)

            tempo_loss = (tempo_loss + delta_loss * self.DELTA_WEIGHT) / (1 + self.DELTA_WEIGHT)

        return tempo_loss

    def handle_data_in_tensor(self, x, y, hierarchy_test=False):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        if self.HIER_MEAS:
            hierarchy_output = y[:, cons.MEAS_TEMPO_IDX:cons.MEAS_TEMPO_IDX + 2]
        elif self.HIER_BEAT:
            hierarchy_output = y[:, cons.BEAT_TEMPO_IDX:cons.BEAT_TEMPO_IDX + 2]

        if hierarchy_test:
            y = y[:, :self.NUM_PRIME_PARAM]
            return x.to(self.DEVICE), (hierarchy_output.to(self.DEVICE), y.to(self.DEVICE))

        if self.HIERARCHY:
            y = hierarchy_output
        elif self.IN_HIER:
            x = torch.cat((x, hierarchy_output), 1)
            y = y[:, :self.NUM_PRIME_PARAM]
        elif self.TRILL:
            x = torch.cat((x, y[:, :self.NUM_PRIME_PARAM]), 1)
            y = y[:, -self.num_trill_param:]
        else:
            y = y[:, :self.NUM_PRIME_PARAM]

        return x.to(self.DEVICE), y.to(self.DEVICE)

    def sigmoid(self, x, gain=1):
        return 1 / (1 + math.exp(-gain * x))


    ### training
    def load_training_data(self):
        print('Loading the training data...')
        training_data_name = self.args.dataName + ".dat"
        training_data_path = f'{self.DATA_DIR}/train/{training_data_name}'
        if not os.path.isfile(training_data_path):
            training_data_name = '/mnt/ssd1/jdasam_data/' + training_data_name
        with open(f'{self.DATA_DIR}/train/{training_data_name}', "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            # p = u.load()
            # complete_xy = pickle.load(f)
            complete_xy = u.load()

        print('Done loading training data')
        return complete_xy

    def train(self, complete_xy):
        # if self.args.sessMode == 'train':
        now = datetime.now()
        current_time = now.strftime("%D %I:%M:%S")
        print(f'Starting training job at {current_time}')
        model_parameters = filter(lambda p: p.requires_grad, self.MODEL.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of Network Parameters is ', params)

        best_prime_loss = float("inf")
        best_second_loss = float("inf")
        best_trill_loss = float("inf")
        start_epoch = 0

        if self.args.resumeTraining and not self.args.trainTrill:
            if os.path.isfile('prime_' + self.args.modelCode + self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.modelCode + self.args.resume))
                # model_codes = ['prime', 'trill']
                filename = 'prime_' + self.args.modelCode + self.args.resume
                checkpoint = torch.load(filename, map_location=self.DEVICE)
                best_valid_loss = checkpoint['best_valid_loss']
                self.MODEL.load_state_dict(checkpoint['state_dict'])
                self.MODEL.device = self.DEVICE
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.NUM_UPDATED = checkpoint['training_step']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epoch']))
                start_epoch = checkpoint['epoch'] - 1
                best_prime_loss = checkpoint['best_valid_loss']
                print('Best valid loss was ', best_prime_loss)

        # load data
        # print('Loading the training data...')
        # training_data_name = self.args.dataName + ".dat"
        # training_data_path = f'{self.DATA_DIR}/train/{training_data_name}'
        # if not os.path.isfile(training_data_path):
        #     training_data_name = '/mnt/ssd1/jdasam_data/' + training_data_name
        # with open(f'{self.DATA_DIR}/train/{training_data_name}', "rb") as f:
        #     u = pickle._Unpickler(f)
        #     u.encoding = 'latin1'
        #     # p = u.load()
        #     # complete_xy = pickle.load(f)
        #     complete_xy = u.load()

        # print('Done loading training data')
        train_xy = complete_xy['train']
        test_xy = complete_xy['valid']
        print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))
        print('training sample example', train_xy[0][0][0])

        train_model = self.MODEL

        # total_step = len(train_loader)
        for epoch in range(start_epoch, self.num_epochs):
            print('current training step is ', self.NUM_UPDATED)
            tempo_loss_total = []
            vel_loss_total = []
            dev_loss_total = []
            articul_loss_total = []
            pedal_loss_total = []
            trill_loss_total = []
            kld_total = []

            if self.RAND_TRAIN:
                num_perf_data = len(train_xy)
                remaining_samples = []
                for i in range(num_perf_data):
                    remaining_samples.append(TraningSample(i))
                while len(remaining_samples) > 0:
                    new_index = random.randrange(0, len(remaining_samples))
                    selected_sample = remaining_samples[new_index]
                    train_x = train_xy[selected_sample.index][0]
                    train_y = train_xy[selected_sample.index][1]
                    if self.args.trainingLoss == 'CE':
                        train_y = self.categorize_value_to_vector(train_y, self.BINS)
                    self.note_locations = train_xy[selected_sample.index][2]
                    align_matched = train_xy[selected_sample.index][3]
                    pedal_status = train_xy[selected_sample.index][4]
                    edges = train_xy[selected_sample.index][5]
                    data_size = len(train_x)

                    if selected_sample.slice_indexes is None:
                        measure_numbers = [x.measure for x in self.note_locations]
                        if self.HIER_MEAS and self.HIERARCHY:
                            selected_sample.slice_indexes = dp.make_slice_with_same_measure_number(data_size,
                                                                                                   measure_numbers,
                                                                                                   measure_steps=self.TIME_STEPS)

                        else:
                            selected_sample.slice_indexes = dp.make_slicing_indexes_by_measure(data_size,
                                                                                               measure_numbers,
                                                                                               steps=self.TIME_STEPS)

                    num_slice = len(selected_sample.slice_indexes)
                    selected_idx = random.randrange(0, num_slice)
                    self.slice_idx = selected_sample.slice_indexes[selected_idx]

                    if self.MODEL.is_graph:
                        graphs = self.edges_to_matrix_short(edges, self.slice_idx)
                    else:
                        graphs = None

                    key_lists = [0]
                    key = 0
                    for i in range(self.num_key_augmentation):
                        while key in key_lists:
                            key = random.randrange(-5, 7)
                        key_lists.append(key)

                    for i in range(self.num_key_augmentation + 1):
                        key = key_lists[i]
                        temp_train_x = dp.key_augmentation(train_x, key)
                        self.kld_weight = self.sigmoid((self.NUM_UPDATED - self.KLD_SIG) / (self.KLD_SIG / 10)) * self.KLD_MAX

                        self.training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                         'note_locations': self.note_locations,
                                         'align_matched': align_matched, 'pedal_status': pedal_status,
                                         'self.slice_idx': self.slice_idx, 'kld_weight': self.kld_weight}

                        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                            self.batch_time_step_run(self.training_data, model=train_model)
                        tempo_loss_total.append(tempo_loss.item())
                        vel_loss_total.append(vel_loss.item())
                        dev_loss_total.append(dev_loss.item())
                        articul_loss_total.append(articul_loss.item())
                        pedal_loss_total.append(pedal_loss.item())
                        trill_loss_total.append(trill_loss.item())
                        kld_total.append(kld.item())
                        self.NUM_UPDATED += 1

                    del selected_sample.slice_indexes[selected_idx]
                    if len(selected_sample.slice_indexes) == 0:
                        # print('every slice in the sample is trained')
                        del remaining_samples[new_index]

            else:
                for xy_tuple in train_xy:
                    train_x = xy_tuple[0]
                    train_y = xy_tuple[1]
                    if self.args.trainingLoss == 'CE':
                        train_y = self.categorize_value_to_vector(train_y, self.BINS)
                    self.note_locations = xy_tuple[2]
                    align_matched = xy_tuple[3]
                    pedal_status = xy_tuple[4]
                    edges = xy_tuple[5]

                    data_size = len(self.note_locations)
                    if self.MODEL.is_graph:
                        graphs = self.edges_to_matrix(edges, data_size)
                    else:
                        graphs = None
                    measure_numbers = [x.measure for x in self.note_locations]
                    # graphs = edges_to_sparse_tensor(edges)
                    total_batch_num = int(math.ceil(data_size / (self.TIME_STEPS * self.batch_size)))

                    key_lists = [0]
                    key = 0
                    for i in range(self.num_key_augmentation):
                        while key in key_lists:
                            key = random.randrange(-5, 7)
                        key_lists.append(key)

                    for i in range(self.num_key_augmentation + 1):
                        key = key_lists[i]
                        temp_train_x = dp.key_augmentation(train_x, key)
                        slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.TIME_STEPS)
                        self.kld_weight = self.sigmoid((self.NUM_UPDATED - self.KLD_SIG) / (self.KLD_SIG / 10)) * self.KLD_MAX

                        for self.slice_idx in slice_indexes:
                            self.training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                             'note_locations': self.note_locations,
                                             'align_matched': align_matched, 'pedal_status': pedal_status,
                                             'self.slice_idx': self.slice_idx, 'kld_weight': self.kld_weight}

                            tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                                self.batch_time_step_run(self.training_data, model=train_model)
                            tempo_loss_total.append(tempo_loss.item())
                            vel_loss_total.append(vel_loss.item())
                            dev_loss_total.append(dev_loss.item())
                            articul_loss_total.append(articul_loss.item())
                            pedal_loss_total.append(pedal_loss.item())
                            trill_loss_total.append(trill_loss.item())
                            kld_total.append(kld.item())
                            self.NUM_UPDATED += 1

            print(
                'Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
                .format(epoch + 1, self.num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
                        np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total),
                        np.mean(trill_loss_total), np.mean(kld_total)))

            ## Validation
            tempo_loss_total = []
            vel_loss_total = []
            deviation_loss_total = []
            articul_loss_total = []
            pedal_loss_total = []
            trill_loss_total = []
            kld_loss_total = []

            for xy_tuple in test_xy:
                test_x = xy_tuple[0]
                test_y = xy_tuple[1]
                self.note_locations = xy_tuple[2]
                align_matched = xy_tuple[3]
                pedal_status = xy_tuple[4]
                edges = xy_tuple[5]
                if self.MODEL.is_graph:
                    graphs = self.edges_to_matrix(edges, len(test_x))
                else:
                    graphs = None
                if self.LOSS_TYPE == 'CE':
                    test_y = self.categorize_value_to_vector(test_y, self.BINS)

                batch_x, batch_y = self.handle_data_in_tensor(test_x, test_y)
                batch_x = batch_x.view(1, -1, self.NUM_INPUT)
                batch_y = batch_y.view(1, -1, self.NUM_OUTPUT)
                # input_y = torch.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(self.DEVICE)
                align_matched = torch.Tensor(align_matched).view(1, -1, 1).to(self.DEVICE)
                pedal_status = torch.Tensor(pedal_status).view(1, -1, 1).to(self.DEVICE)
                outputs, total_z = self.run_model_in_steps(batch_x, batch_y, graphs, self.note_locations)

                # valid_loss = self.criterion(outputs[:,:,self.NUM_TEMPO_PARAM:-num_trill_param], batch_y[:,:,self.NUM_TEMPO_PARAM:-num_trill_param], align_matched)
                if self.HIERARCHY:
                    if self.HIER_MEAS:
                        hierarchy_numbers = [x.measure for x in self.note_locations]
                    elif self.HIER_BEAT:
                        hierarchy_numbers = [x.beat for x in self.note_locations]
                    tempo_y = self.MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 0)
                    vel_y = self.MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 1)

                    tempo_loss = self.criterion(outputs[:, :, 0:1], tempo_y)
                    vel_loss = self.criterion(outputs[:, :, 1:2], vel_y)
                    if self.args.deltaLoss:
                        tempo_out_delta = outputs[:, 1:, 0:1] - outputs[:, :-1, 0:1]
                        tempo_true_delta = tempo_y[:, 1:, :] - tempo_y[:, :-1, :]
                        vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
                        vel_true_delta = vel_y[:, 1:, :] - vel_y[:, :-1, :]

                        tempo_loss += self.criterion(tempo_out_delta, tempo_true_delta) * self.DELTA_WEIGHT
                        vel_loss += self.criterion(vel_out_delta, vel_true_delta) * self.DELTA_WEIGHT

                    deviation_loss = torch.zeros(1)
                    articul_loss = torch.zeros(1)
                    pedal_loss = torch.zeros(1)
                    trill_loss = torch.zeros(1)

                    for z in total_z:
                        perform_mu, perform_var = z
                        kld_loss = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                        kld_loss_total.append(kld_loss.item())
                elif self.TRILL:
                    trill_bool = batch_x[:, :, self.is_trill_index_concated] == 1
                    trill_bool = trill_bool.float().view(1, -1, 1).to(self.DEVICE)
                    trill_loss = self.criterion(outputs, batch_y, trill_bool)

                    tempo_loss = torch.zeros(1)
                    vel_loss = torch.zeros(1)
                    deviation_loss = torch.zeros(1)
                    articul_loss = torch.zeros(1)
                    pedal_loss = torch.zeros(1)
                    kld_loss = torch.zeros(1)
                    kld_loss_total.append(kld_loss.item())

                else:
                    tempo_loss = self.cal_tempo_loss_in_beat(outputs, batch_y, self.note_locations, 0)
                    if self.LOSS_TYPE == 'CE':
                        vel_loss = self.criterion(outputs[:, :, self.NUM_TEMPO_PARAM:self.NUM_TEMPO_PARAM + len(self.BINS[1])],
                                             batch_y[:, :, self.NUM_TEMPO_PARAM:self.NUM_TEMPO_PARAM + len(self.BINS[1])],
                                             align_matched)
                        deviation_loss = self.criterion(
                            outputs[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]):self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2])],
                            batch_y[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]):self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2])])
                        pedal_loss = self.criterion(
                            outputs[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2]):-self.num_trill_param],
                            batch_y[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2]):-self.num_trill_param])
                        trill_loss = self.criterion(outputs[:, :, -self.num_trill_param:], batch_y[:, :, -self.num_trill_param:])
                    else:
                        vel_loss = self.criterion(outputs[:, :, self.VEL_PARAM_IDX], batch_y[:, :, self.VEL_PARAM_IDX], align_matched)
                        deviation_loss = self.criterion(outputs[:, :, self.DEV_PARAM_IDX], batch_y[:, :, self.DEV_PARAM_IDX],
                                                   align_matched)
                        articul_loss = self.criterion(outputs[:, :, self.PEDAL_PARAM_IDX], batch_y[:, :, self.PEDAL_PARAM_IDX],
                                                 pedal_status)
                        pedal_loss = self.criterion(outputs[:, :, self.PEDAL_PARAM_IDX + 1:], batch_y[:, :, self.PEDAL_PARAM_IDX + 1:],
                                               align_matched)
                        trill_loss = torch.zeros(1)
                    for z in total_z:
                        perform_mu, perform_var = z
                        kld_loss = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                        kld_loss_total.append(kld_loss.item())

                # valid_loss_total.append(valid_loss.item())
                tempo_loss_total.append(tempo_loss.item())
                vel_loss_total.append(vel_loss.item())
                deviation_loss_total.append(deviation_loss.item())
                articul_loss_total.append(articul_loss.item())
                pedal_loss_total.append(pedal_loss.item())
                trill_loss_total.append(trill_loss.item())

            mean_tempo_loss = np.mean(tempo_loss_total)
            mean_vel_loss = np.mean(vel_loss_total)
            mean_deviation_loss = np.mean(deviation_loss_total)
            mean_articul_loss = np.mean(articul_loss_total)
            mean_pedal_loss = np.mean(pedal_loss_total)
            mean_trill_loss = np.mean(trill_loss_total)
            mean_kld_loss = np.mean(kld_loss_total)

            mean_valid_loss = (
                                          mean_tempo_loss + mean_vel_loss + mean_deviation_loss + mean_articul_loss + mean_pedal_loss * 7 + mean_kld_loss * self.kld_weight) / (
                                          11 + self.kld_weight)

            print(
                "Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
                .format(mean_valid_loss, mean_tempo_loss, mean_vel_loss,
                        mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss))

            is_best = mean_valid_loss < best_prime_loss
            best_prime_loss = min(mean_valid_loss, best_prime_loss)

            is_best_trill = mean_trill_loss < best_trill_loss
            best_trill_loss = min(mean_trill_loss, best_trill_loss)

            if self.args.trainTrill:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.MODEL.state_dict(),
                    'best_valid_loss': best_trill_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'training_step': self.NUM_UPDATED
                }, is_best_trill, model_name='trill')
            else:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.MODEL.state_dict(),
                    'best_valid_loss': best_prime_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'training_step': self.NUM_UPDATED
                }, is_best, model_name='prime')

        # end of epoch

    def testEvaluation(self):
        # elif self.args.sessMode in ['test', 'testAll', 'testAllzero', 'encode', 'encodeAll', 'evaluate', 'correlation']:
        ### test session
        if os.path.isfile('prime_' + self.args.modelCode + self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.modelCode + self.args.resume))
            # model_codes = ['prime', 'trill']
            filename = 'prime_' + self.args.modelCode + self.args.resume
            print('device is ', self.args.device)
            torch.cuda.set_device(self.args.device)
            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'
            checkpoint = torch.load(filename, map_location=map_location)
            # self.args.start_epoch = checkpoint['epoch']
            # best_valid_loss = checkpoint['best_valid_loss']
            self.MODEL.load_state_dict(checkpoint['state_dict'])
            # self.MODEL.num_graph_iteration = 10
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            # self.NUM_UPDATED = checkpoint['training_step']
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # trill_filename = self.args.trillCode + self.args.resume
            trill_filename = self.args.trillCode + '_best.pth.tar'
            checkpoint = torch.load(trill_filename, map_location=map_location)
            self.TRILL_MODEL.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(trill_filename, checkpoint['epoch']))

            if self.IN_HIER:
                self.HIER_MODEL_PARAM = param.load_parameters(self.args.hierCode + '_param')
                self.HIER_MODEL = nnModel.HAN_Integrated(self.HIER_MODEL_PARAM, self.DEVICE, True).to(self.DEVICE)
                filename = 'prime_' + self.args.hierCode + self.args.resume
                checkpoint = torch.load(filename, map_location=self.DEVICE)
                self.HIER_MODEL.load_state_dict(checkpoint['state_dict'])
                print("=> high-level model loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))
        self.MODEL.is_teacher_force = False

        if self.args.sessMode == 'test':
            random.seed(0)
            self.load_file_and_generate_performance(self.args.testPath)
        elif self.args.sessMode == 'testAll':
            path_list = cons.emotion_data_path
            emotion_list = cons.emotion_key_list
            perform_z_by_list = self.encode_all_emotionNet_data(path_list, emotion_list)
            test_list = cons.test_piece_list
            print(f'Test pieces: {test_list}')
            for piece in test_list:
                path = f'{self.DATA_DIR}/test_pieces/{piece[0]}/'
                # path = './data/test_pieces/' + piece[0] + '/'
                composer = piece[1]
                if len(piece) == 3:
                    start_tempo = piece[2]
                else:
                    start_tempo = 0
                for perform_z_pair in perform_z_by_list:
                    self.load_file_and_generate_performance(path, composer, z=perform_z_pair, start_tempo=start_tempo)
                self.load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)
        elif self.args.sessMode == 'testAllzero':
            test_list = cons.test_piece_list
            for piece in test_list:
                path = f'{self.DATA_DIR}/test_pieces/{piece[0]}/'
                # path = './test_pieces/' + piece[0] + '/'
                composer = piece[1]
                if len(piece) == 3:
                    start_tempo = piece[2]
                else:
                    start_tempo = 0
                random.seed(0)
                self.load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)

        elif self.args.sessMode == 'encode':
            perform_z, qpm_primo = self.load_file_and_encode_style(self.args.testPath, self.args.perfName, self.args.composer)
            print(perform_z)
            with open(self.args.testPath + self.args.perfName + '_style' + '.dat', 'wb') as f:
                pickle.dump(perform_z, f, protocol=2)

        elif self.args.sessMode == 'evaluate':
            test_data_name = self.args.dataName + "_test.dat"
            if not os.path.isfile(test_data_name):
                test_data_name = '/mnt/ssd1/jdasam_data/' + test_data_name
            with open(f'{self.DATA_DIR}/test/{test_data_name}', "rb") as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                # p = u.load()
                # complete_xy = pickle.load(f)
                complete_xy = u.load()

            tempo_loss_total = []
            vel_loss_total = []
            deviation_loss_total = []
            trill_loss_total = []
            articul_loss_total = []
            pedal_loss_total = []
            kld_total = []

            prev_perf_x = complete_xy[0][0]
            prev_perfs_worm_data = []
            prev_reconstructed_worm_data = []
            prev_zero_predicted_worm_data = []
            piece_wise_loss = []
            human_correlation_total = []
            human_correlation_results = xml_matching.CorrelationResult()
            model_correlation_total = []
            model_correlation_results = xml_matching.CorrelationResult()
            zero_sample_correlation_total = []
            zero_sample_correlation_results = xml_matching.CorrelationResult()

            for xy_tuple in complete_xy:
                current_perf_index = complete_xy.index(xy_tuple)
                test_x = xy_tuple[0]
                test_y = xy_tuple[1]
                self.note_locations = xy_tuple[2]
                align_matched = xy_tuple[3]
                pedal_status = xy_tuple[4]
                edges = xy_tuple[5]
                graphs = self.edges_to_matrix(edges, len(test_x))
                if self.LOSS_TYPE == 'CE':
                    test_y = self.categorize_value_to_vector(test_y, self.BINS)

                if xml_matching.check_feature_pair_is_from_same_piece(prev_perf_x, test_x):
                    piece_changed = False
                    # current_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(test_y, self.note_locations=self.note_locations, momentum=0.2)
                    # for prev_worm in prev_perfs_worm_data:
                    #     tempo_r, _ = xml_matching.cal_correlation(current_perf_worm_data[0], prev_worm[0])
                    #     dynamic_r, _ = xml_matching.cal_correlation(current_perf_worm_data[1], prev_worm[1])
                    #     human_correlation_results.append_result(tempo_r, dynamic_r)
                    # prev_perfs_worm_data.append(current_perf_worm_data)
                else:
                    piece_changed = True

                if piece_changed or current_perf_index == len(complete_xy) - 1:
                    prev_perf_x = test_x
                    if piece_wise_loss:
                        piece_wise_loss_mean = np.mean(np.asarray(piece_wise_loss), axis=0)
                        tempo_loss_total.append(piece_wise_loss_mean[0])
                        vel_loss_total.append(piece_wise_loss_mean[1])
                        deviation_loss_total.append(piece_wise_loss_mean[2])
                        articul_loss_total.append(piece_wise_loss_mean[3])
                        pedal_loss_total.append(piece_wise_loss_mean[4])
                        trill_loss_total.append(piece_wise_loss_mean[5])
                        kld_total.append(piece_wise_loss_mean[6])
                    piece_wise_loss = []

                    # human_correlation_total.append(human_correlation_results)
                    # human_correlation_results = xml_matching.CorrelationResult()
                    #
                    # for predict in prev_reconstructed_worm_data:
                    #     for human in prev_perfs_worm_data:
                    #         tempo_r, _ = xml_matching.cal_correlation(predict[0], human[0])
                    #         dynamic_r, _ = xml_matching.cal_correlation(predict[1], human[1])
                    #         model_correlation_results.append_result(tempo_r, dynamic_r)
                    #
                    # model_correlation_total.append(model_correlation_results)
                    # model_correlation_results = xml_matching.CorrelationResult()
                    #
                    # for zero in prev_zero_predicted_worm_data:
                    #     for human in prev_perfs_worm_data:
                    #         tempo_r, _ = xml_matching.cal_correlation(zero[0], human[0])
                    #         dynamic_r, _ = xml_matching.cal_correlation(zero[1], human[1])
                    #         zero_sample_correlation_results.append_result(tempo_r, dynamic_r)
                    #
                    # zero_sample_correlation_total.append(zero_sample_correlation_results)
                    # zero_sample_correlation_results = xml_matching.CorrelationResult()
                    #
                    # prev_reconstructed_worm_data = []
                    # prev_zero_predicted_worm_data = []
                    # prev_perfs_worm_data = []
                    #
                    # print('Human Correlation: ', human_correlation_total[-1])
                    # print('Reconst Correlation: ', model_correlation_total[-1])
                    # print('Zero Sampled Correlation: ', zero_sample_correlation_total[-1])

                batch_x, batch_y = self.handle_data_in_tensor(test_x, test_y, hierarchy_test=self.IN_HIER)
                align_matched = torch.Tensor(align_matched).view(1, -1, 1).to(self.DEVICE)
                pedal_status = torch.Tensor(pedal_status).view(1, -1, 1).to(self.DEVICE)

                if self.IN_HIER:
                    batch_x = batch_x.view((1, -1, self.HIER_MODEL.input_size))
                    hier_y = batch_y[0].view(1, -1, self.HIER_MODEL.output_size)
                    hier_outputs, _ = self.run_model_in_steps(batch_x, hier_y, graphs, self.note_locations, model=self.HIER_MODEL)
                    if self.HIER_MEAS:
                        hierarchy_numbers = [x.measure for x in self.note_locations]
                    elif self.HIER_BEAT:
                        hierarchy_numbers = [x.beat for x in self.note_locations]
                    hier_outputs_spanned = self.HIER_MODEL.span_beat_to_note_num(hier_outputs, hierarchy_numbers,
                                                                                 batch_x.shape[1], 0)
                    input_concat = torch.cat((batch_x, hier_outputs_spanned), 2)
                    batch_y = batch_y[1].view(1, -1, self.MODEL.output_size)
                    outputs, perform_z = self.run_model_in_steps(input_concat, batch_y, graphs, self.note_locations,
                                                            model=self.MODEL)

                    # make another prediction with random sampled z
                    zero_hier_outputs, _ = self.run_model_in_steps(batch_x, hier_y, graphs, self.note_locations,
                                                              model=self.HIER_MODEL,
                                                              initial_z='zero')
                    zero_hier_spanned = self.HIER_MODEL.span_beat_to_note_num(zero_hier_outputs, hierarchy_numbers,
                                                                              batch_x.shape[1], 0)
                    zero_input_concat = torch.cat((batch_x, zero_hier_spanned), 2)
                    zero_prediction, _ = self.run_model_in_steps(zero_input_concat, batch_y, graphs, self.note_locations,
                                                            model=self.MODEL)

                else:
                    batch_x = batch_x.view((1, -1, self.NUM_INPUT))
                    batch_y = batch_y.view((1, -1, self.NUM_OUTPUT))
                    outputs, perform_z = self.run_model_in_steps(batch_x, batch_y, graphs, self.note_locations)

                    # make another prediction with random sampled z
                    zero_prediction, _ = self.run_model_in_steps(batch_x, batch_y, graphs, self.note_locations, model=self.MODEL,
                                                            initial_z='zero')

                output_as_feature = outputs.view(-1, self.NUM_OUTPUT).cpu().numpy()
                predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(output_as_feature, self.note_locations,
                                                                                    momentum=0.2)
                zero_prediction_as_feature = zero_prediction.view(-1, self.NUM_OUTPUT).cpu().numpy()
                zero_predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(zero_prediction_as_feature,
                                                                                         self.note_locations,
                                                                                         momentum=0.2)

                prev_reconstructed_worm_data.append(predicted_perf_worm_data)
                prev_zero_predicted_worm_data.append(zero_predicted_perf_worm_data)

                # for prev_worm in prev_perfs_worm_data:
                #     tempo_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[0], prev_worm[0])
                #     dynamic_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[1], prev_worm[1])
                #     model_correlation_results.append_result(tempo_r, dynamic_r)
                # print('Model Correlation: ', model_correlation_results)

                # valid_loss = self.criterion(outputs[:,:,self.NUM_TEMPO_PARAM:-self.num_trill_param], batch_y[:,:,self.NUM_TEMPO_PARAM:-self.num_trill_param], align_matched)
                if self.MODEL.is_baseline:
                    tempo_loss = self.criterion(outputs[:, :, 0], batch_y[:, :, 0], align_matched)
                else:
                    tempo_loss = self.cal_tempo_loss_in_beat(outputs, batch_y, self.note_locations, 0)
                if self.LOSS_TYPE == 'CE':
                    vel_loss = self.criterion(outputs[:, :, self.NUM_TEMPO_PARAM:self.NUM_TEMPO_PARAM + len(self.BINS[1])],
                                         batch_y[:, :, self.NUM_TEMPO_PARAM:self.NUM_TEMPO_PARAM + len(self.BINS[1])], align_matched)
                    deviation_loss = self.criterion(
                        outputs[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]):self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2])],
                        batch_y[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]):self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2])])
                    pedal_loss = self.criterion(
                        outputs[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2]):-self.num_trill_param],
                        batch_y[:, :, self.NUM_TEMPO_PARAM + len(self.BINS[1]) + len(self.BINS[2]):-self.num_trill_param])
                    trill_loss = self.criterion(outputs[:, :, -self.num_trill_param:], batch_y[:, :, -self.num_trill_param:])
                else:
                    vel_loss = self.criterion(outputs[:, :, self.VEL_PARAM_IDX], batch_y[:, :, self.VEL_PARAM_IDX], align_matched)
                    deviation_loss = self.criterion(outputs[:, :, self.DEV_PARAM_IDX], batch_y[:, :, self.DEV_PARAM_IDX],
                                               align_matched)
                    articul_loss = self.criterion(outputs[:, :, self.PEDAL_PARAM_IDX], batch_y[:, :, self.PEDAL_PARAM_IDX],
                                             pedal_status)
                    pedal_loss = self.criterion(outputs[:, :, self.PEDAL_PARAM_IDX + 1:], batch_y[:, :, self.PEDAL_PARAM_IDX + 1:],
                                           align_matched)
                    trill_loss = torch.zeros(1)

                piece_kld = []
                for z in perform_z:
                    perform_mu, perform_var = z
                    kld = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    piece_kld.append(kld)
                piece_kld = torch.mean(torch.stack(piece_kld))

                piece_wise_loss.append((tempo_loss.item(), vel_loss.item(), deviation_loss.item(), articul_loss.item(),
                                        pedal_loss.item(), trill_loss.item(), piece_kld.item()))

            mean_tempo_loss = np.mean(tempo_loss_total)
            mean_vel_loss = np.mean(vel_loss_total)
            mean_deviation_loss = np.mean(deviation_loss_total)
            mean_articul_loss = np.mean(articul_loss_total)
            mean_pedal_loss = np.mean(pedal_loss_total)
            mean_trill_loss = np.mean(trill_loss_total)
            mean_kld_loss = np.mean(kld_total)

            mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss / 2 + mean_pedal_loss * 8) / 10.5

            print(
                "Test Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}"
                .format(mean_valid_loss, mean_tempo_loss, mean_vel_loss,
                        mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss, mean_kld_loss))
            # num_piece = len(model_correlation_total)
            # for i in range(num_piece):
            #     if len(human_correlation_total) > 0:
            #         print('Human Correlation: ', human_correlation_total[i])
            #         print('Model Correlation: ', model_correlation_total[i])


        elif self.args.sessMode == 'correlation':
            with open('selected_corr_30.dat', "rb") as f:
                u = pickle._Unpickler(f)
                selected_corr = u.load()
            model_cor = []
            for piece_corr in selected_corr:
                if piece_corr is None or piece_corr == []:
                    continue
                path = piece_corr[0].path_name
                composer_name = copy.copy(path).split('/')[1]
                output_features = self.load_file_and_generate_performance(path, composer_name, 'zero', return_features=True)
                for slice_corr in piece_corr:
                    slc_idx = slice_corr.slice_index
                    sliced_features = output_features[slc_idx[0]:slc_idx[1]]
                    tempos, dynamics = perf_worm.cal_tempo_and_velocity_by_beat(sliced_features)
                    model_correlation_results = xml_matching.CorrelationResult()
                    model_correlation_results.path_name = slice_corr.path_name
                    model_correlation_results.slice_index = slice_corr.slice_index
                    human_tempos = slice_corr.tempo_features
                    human_dynamics = slice_corr.dynamic_features
                    for i in range(slice_corr.num_performance):
                        tempo_r, _ = xml_matching.cal_correlation(tempos, human_tempos[i])
                        dynamic_r, _ = xml_matching.cal_correlation(dynamics, human_dynamics[i])
                        model_correlation_results._append_result(tempo_r, dynamic_r)
                    print(model_correlation_results)
                    model_correlation_results.tempo_features = copy.copy(slice_corr.tempo_features)
                    model_correlation_results.dynamic_features = copy.copy(slice_corr.dynamic_features)
                    model_correlation_results.tempo_features.append(tempos)
                    model_correlation_results.dynamic_features.append(dynamics)

                    save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_',
                                                                                           10) + '_note{}-{}_by_{}.png'.format(
                        slc_idx[0], slc_idx[1], self.args.modelCode)
                    perf_worm.plot_human_model_features_compare(model_correlation_results.tempo_features, save_name)
                    model_cor.append(model_correlation_results)

            with open(self.args.modelCode + "_cor.dat", "wb") as f:
                pickle.dump(model_cor, f, protocol=2)


class TraningSample():
    def __init__(self, index):
        self.index = index
        self.slice_indexes = None
