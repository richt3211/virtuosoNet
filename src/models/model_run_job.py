from math import nan
import math
from time import time
from src.logger import init_logger
from src.constants import CACHE_MODEL_DIR
from src.discord_bot import sendToDiscord
from src.models.model_writer_reader import save_checkpoint, save_params
from src.models.params import Params
from src.neptune import log_neptune_timeline
from dataclasses import asdict, dataclass
from neptune.experiments import Experiment
from datetime import datetime

import src.old.data_process as dp
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
import shutil
import torch
import random 
import copy
import logging
import pickle


@dataclass
class ModelJobParams(Params):
    qpm_index:int = 0
    vel_param_idx:int = 1
    dev_param_idx:int = 2
    articul_param_idx:int = 3
    pedal_param_idx:int = 4

    num_key_augmentation:int = 1
    batch_size:int = 1
    epochs:int = 20

    num_tempo_param:int = 1
    num_prime_param:int = 11

    criterion:str = 'torch'
    tempo_loss:bool = True

    articul_mask:str = 'aligned'

    tempo_weight:float = 0.2
    vel_weight:float = 0.2
    dev_weight:float = 0.2
    articul_weight:float = 0.2
    pedal_weight:float = 0.2

class ModelJob():

    def __init__(self, params:ModelJobParams, model:nn.Module, exp:Experiment):
        self.params = params
        self.exp = exp
        self.feature_loss_init = {
            'tempo': [],
            'vel': [],
            'dev': [],
            'articul': [],
            'pedal': [],
            'trill': [],
            'kld': []
        }
        self.model_name = None
        self.hyper_params = None
        self.model = model
        self.num_updated = 0

    def run_job(self, data, model_folder):
        try:
            type = "DEV" if self.params.is_dev else ""
            start_message = f"STARTING {self.model_name} JOB AT {self.params.epochs} EPOCHS FOR {type} DATA SET"
            log_neptune_timeline(start_message, self.exp)
            logging.info(start_message)
            sendToDiscord(start_message)

            model_params = f'Number of model params: {self.count_paramters()}'
            self.exp.log_text('number of model params', model_params)
            logging.info(model_params)

            architecture = repr(self.model)
            self.exp.log_text('model architecture', architecture)
            logging.info(architecture)

            # training_loss_total, valid_loss_total = self.train(self.model, data, version, model_folder)
            self.train(self.model, data, model_folder)

            end_message = f'FINISHED {self.model_name} TRAINING JOB AT {self.params.epochs} EPOCHS FOR {type} DATA SET'
            log_neptune_timeline(end_message, self.exp)
            logging.info(end_message)
            sendToDiscord(end_message)
            # return training_loss_total, valid_loss_total
            self.exp.stop()

        except Exception as e:
            self.exp.log_text('error message', 'Error in training, stopping')
            self.exp.log_text('error message', str(e))
            self.exp.stop(str(e))
            sendToDiscord("There was an error during training for the HAN BL training job, please check logs")
            raise e

    def train(self, model, data, model_folder):
        best_loss = float('inf')
        for epoch in range(self.params.epochs):
            epoch_num = epoch +1
            message = f'Training Epoch {epoch_num}'
            log_neptune_timeline(message, self.exp)
            logging.info(message)

            training_loss, training_feature_loss = self.train_epoch(model, data['train'])
            self.log_loss(training_loss, training_feature_loss, 'train', epoch_num)

            valid_loss, valid_feature_loss = self.evaluate(model, data['valid'])
            self.log_loss(valid_loss, valid_feature_loss, 'valid', epoch_num)

            mean_valid_loss = np.mean(valid_loss)
            # if ((epoch_num) >= 5 and (epoch_num) % 5 == 0):
            log_message = f'Trained model for {epoch_num} epochs with validation loss of {mean_valid_loss}'
            logging.info(log_message)
            log_neptune_timeline(log_message, self.exp)
            sendToDiscord(log_message)

            is_best = mean_valid_loss < best_loss
            best_loss = min(mean_valid_loss, best_loss)

            save_checkpoint(
                state={
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_valid_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'training_step': self.num_updated
                }, 
                is_best=is_best, 
                is_dev=self.params.is_dev,
                exp=self.exp,
            )
            save_params(model_folder, self.model.params, self.exp)

            message = f'saving model at epoch {epoch +1} as the best model'
            logging.info(message)
            log_neptune_timeline(message, self.exp)
            self.exp.log_metric('trained epochs', epoch + 1)

    def train_epoch(self, model, train_data):
        self.init_optimizer(model)
        model.train()
        feature_loss = copy.deepcopy(self.feature_loss_init)
        total_loss = []
        for xy_tuple in train_data:
            train_x = xy_tuple[0]
            train_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]

            data_size = len(note_locations)
            measure_numbers = [x.measure for x in note_locations]

            key_lists = [0]
            key = 0
            for i in range(self.params.num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(self.params.num_key_augmentation+1):
                key = key_lists[i]
                temp_train_x = dp.key_augmentation(train_x, key)
                slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.params.time_steps)

                training_data = {
                    'x': temp_train_x, 
                    'y': train_y,
                    'note_locations': note_locations,
                    'align_matched': align_matched, 
                    'pedal_status': pedal_status,
                    'slice_indexes': slice_indexes
                }
                self.run_for_performance(training_data, model, feature_loss, total_loss, train=True)
                # print(feature_loss)
                # print(total_loss)
        return total_loss, feature_loss

    def evaluate(self, model, eval_data):
        model.eval()
        feature_loss = copy.deepcopy(self.feature_loss_init)
        total_loss = []
        with torch.no_grad():
            for xy_tuple in eval_data:
                eval_x = xy_tuple[0]
                eval_y = xy_tuple[1]
                note_locations = xy_tuple[2] 
                align_matched = xy_tuple[3]
                pedal_status = xy_tuple[4]

                data_size = len(note_locations)
                measure_numbers = [x.measure for x in note_locations]

                slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.params.time_steps)
                eval_data = {
                    'x': eval_x, 
                    'y': eval_y,
                    'note_locations': note_locations,
                    'align_matched': align_matched, 
                    'pedal_status': pedal_status,
                    'slice_indexes': slice_indexes
                }
                self.run_for_performance(eval_data, model, feature_loss, total_loss, train=False)
        return total_loss, feature_loss

    def run_for_performance(self, data, model, feature_loss, total_loss, train=True):
        for slice_idx in data['slice_indexes']:
            data['slice_idx'] = slice_idx
            self.batch_time_step_run(data, model, feature_loss, total_loss, train)
            # print(feature_loss)
            # print(total_loss)
            self.num_updated += 1



    def get_batch_and_alignment(self, data):
        batch_start, batch_end = data['slice_idx']
        batch_x, batch_y = self.handle_data_in_tensor(data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

        # batch_x_ = batch_x.view((self.params.batch_size, -1, self.params.input_size))
        # batch_y_ = batch_y.view((self.params.batch_size, -1, self.params.output_size))

        batch_x = batch_x.view((-1, self.params.batch_size, self.params.input_size))
        batch_y = batch_y.view((-1, self.params.batch_size, self.params.output_size))

        # align_matched_ = torch.Tensor(data['align_matched'][batch_start:batch_end]).view((self.params.batch_size, -1, 1)).to(self.params.
        # device)
        # pedal_status_ = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view((self.params.batch_size, -1, 1)).to(self.params.device)

        # note_locations = torch.Tensor(data['note_locations'][batch_start:batch_end]).view((-1, self.params.batch_size, 1)).to(self.params.device)
        align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end]).view((-1, self.params.batch_size, 1)).to(self.params.device)
        pedal_status = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view((-1, self.params.batch_size, 1)).to(self.params.device)


        prime_batch_x = batch_x
        prime_batch_y = batch_y[:, :, 0:self.params.num_prime_param]

        return prime_batch_x, prime_batch_y, data['note_locations'], align_matched, pedal_status, batch_start
        # return {
        #     'batch_x': prime_batch_x,
        #     'batch_y': prime_batch_y,
        #     'align_matched': align_matched,
        #     'pedal_status': pedal_status
        # }

    def batch_time_step_run(self, data, model, feature_loss, loss, train=True):
        batch_x, batch_y, note_locations, align_matched, pedal_status, batch_start = self.get_batch_and_alignment(data)

        self.zero_grad_optim()
        outputs = model(batch_x)
        # print(outputs)
        # print(outputs.shape)
        # print(outputs[0][0])

        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, total_loss = self.calculate_loss(outputs, batch_y, note_locations, align_matched, pedal_status, batch_start)
        if math.isnan(total_loss):
            print(f'total loss is nan at time step {self.num_updated}')
            print('model output tensor')
            print(outputs.shape)
            print(outputs)
            print(outputs[0][0][0:10])
            exit()

        if train:
            self.step_optimizer(model, total_loss)

        # print(tempo_loss)
        feature_loss['tempo'].append(tempo_loss.item())
        feature_loss['vel'].append(vel_loss.item())
        feature_loss['dev'].append(dev_loss.item())
        feature_loss['articul'].append(articul_loss.item())
        feature_loss['pedal'].append(pedal_loss.item())
        feature_loss['trill'].append(trill_loss.item())

        loss.append(total_loss.item())

        return outputs

    def calculate_loss(self, outputs, batch_y, note_locations, align_matched, pedal_status, batch_start):
        if not self.params.tempo_loss:
            tempo_loss = self.criterion(
                outputs[:, :, 0:1], 
                batch_y[:, :, 0:1], 
                align_matched
            )
        else:
            tempo_loss = self.cal_tempo_loss_in_beat(outputs, batch_y, note_locations, batch_start)
        vel_loss = self.criterion(
            outputs[:, :, self.params.vel_param_idx:self.params.dev_param_idx], 
            batch_y[:, :, self.params.vel_param_idx:self.params.dev_param_idx], 
            align_matched
        )
        dev_loss = self.criterion(
            outputs[:, :, self.params.dev_param_idx:self.params.articul_param_idx], 
            batch_y[:, :, self.params.dev_param_idx:self.params.articul_param_idx], 
            align_matched
        )
        # for some reason the alignment passed in for articul was the pedal status, as opposed to the alignment. 
        # this means that articulation is only calculated for notes that have the sustain pedal pressed, which is what 
        # we don't want. 
        articul_alignment = align_matched if self.params.articul_mask == 'aligned' else pedal_status
        articul_loss = self.criterion(
            outputs[:, :, self.params.articul_param_idx:self.params.pedal_param_idx], 
            batch_y[:, :, self.params.articul_param_idx:self.params.pedal_param_idx], 
            articul_alignment
        )
        pedal_loss = self.criterion(
            outputs[:, :, self.params.pedal_param_idx:], 
            batch_y[:, :, self.params.pedal_param_idx:], 
            align_matched
        )
        #TODO: Experiment with weighted loss calcuation. 
        total_weight = self.params.tempo_weight + self.params.vel_weight + self.params.dev_weight + self.params.articul_weight + self.params.pedal_weight
        total_loss = (
            self.params.tempo_weight * tempo_loss + 
            self.params.vel_weight * vel_loss + 
            self.params.dev_weight * dev_loss + 
            self.params.articul_weight * articul_loss +
            self.params.pedal_weight * pedal_loss 
        ) / total_weight

        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), total_loss

    def cal_tempo_loss_in_beat(self, pred_x, true_x, note_locations, start_index):
        previous_beat = -1

        num_notes = pred_x.shape[1]
        start_beat = note_locations[start_index].beat
        num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1

        pred_beat_tempo = torch.zeros([num_beats, self.params.num_tempo_param]).to(self.params.device)
        true_beat_tempo = torch.zeros([num_beats, self.params.num_tempo_param]).to(self.params.device)
        for i in range(num_notes):
            current_beat = note_locations[i+start_index].beat
            if current_beat > previous_beat:
                previous_beat = current_beat
                # if 'baseline' in args.modelCode:
                #     for j in range(i, num_notes):
                #         if note_locations[j+start_index].beat > current_beat:
                #             break
                #     if not i == j:
                #         pred_beat_tempo[current_beat - start_beat] = torch.mean(pred_x[0, i:j, QPM_INDEX])
                #         true_beat_tempo[current_beat - start_beat] = torch.mean(true_x[0, i:j, QPM_INDEX])
                # else:
                pred_beat_tempo[current_beat-start_beat] = pred_x[0,i,self.params.qpm_index:self.params.qpm_index + self.params.num_tempo_param]
                true_beat_tempo[current_beat-start_beat] = true_x[0,i,self.params.qpm_index:self.params.qpm_index + self.params.num_tempo_param]

        tempo_loss = self.criterion(pred_beat_tempo, true_beat_tempo)
        # if args.deltaLoss and pred_beat_tempo.shape[0] > 1:
        #     prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
        #     true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
        #     delta_loss = criterion(prediction_delta, true_delta)

        #     tempo_loss = (tempo_loss + delta_loss * DELTA_WEIGHT) / (1 + DELTA_WEIGHT)

        return tempo_loss

    def handle_data_in_tensor(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        y = y[:, :self.params.num_prime_param]

        return x.to(self.params.device), y.to(self.params.device)

    def log_loss(self, total_loss: list, feature_loss, type:str, epoch:int):
        self.exp.log_metric(f'{type} total loss', epoch, np.mean(total_loss))
        log_str = f'{type} total loss: {np.mean(total_loss)}'
        for key, value in feature_loss.items():
            self.exp.log_metric(f'{type} {key} loss', epoch, np.mean(value))
            log_str += f'{key} loss: {np.mean(value)}, '
        logging.info(log_str)

    def save_checkpoint(self, state, is_best, folder):
        '''Saves the version of the model at each epoch, and updates the best version 
        of the model'''

        folder = f'{CACHE_MODEL_DIR}/{folder}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = ''
        if self.params.is_dev:
            file_name = f'model_dev.pth'
        else:
            file_name = f'model.pth'
        filepath = f'{folder}/{file_name}'
        torch.save(state, filepath)
        self.exp.log_artifact(filepath, file_name)
        if is_best:
            file_name = ''
            if self.params.is_dev:
                file_name = f'model_dev_best.pth'
            else:
                file_name = f'model_best.pth'

            best_filepath = f'{folder}/{file_name}'
            shutil.copyfile(filepath, best_filepath)
            self.exp.log_artifact(best_filepath, file_name)

    def criterion(self, pred, target, aligned_status=1):
        if self.params.criterion == 'torch':
            return self.torch_criterion(pred, target, aligned_status)
        elif self.params.criterion == 'han':
            return self.han_criterion(pred, target, aligned_status)
        else:
            raise Exception('Invalid criterion choice')

    def torch_criterion(self, pred, target, aligned_status=1):
        loss = nn.MSELoss()
        if isinstance(aligned_status, int):
            output = loss(pred, target)
        else:
            # create a boolean mask all of the notes that aren't aligned
            status_squeezed = torch.squeeze(aligned_status)
            mask = status_squeezed == 1
            # check if mask is completely empty
            all_false = mask.byte().any().item() == 0
            if all_false:
                log_neptune_timeline("No matching notes in batch", self.exp)
                return torch.tensor(0)
            # index pred and target only by notes that are aligned
            p = pred[mask]
            t = target[mask]
            output = loss(p, t)
            if math.isnan(output):
                print(mask)
                print(f'loss is nan at step: {self.num_updated}')
                print(output)
                print('pred')
                print(p[:5])
                print('target')
                print(t[:5])

        return output

    def han_criterion(self, pred, target, aligned_status=1):
        if isinstance(aligned_status, int):
            data_size = pred.shape[-2] * pred.shape[-1]
        else:
            data_size = torch.sum(aligned_status).item() * pred.shape[-1]
            if data_size == 0:
                data_size = 1
        if target.shape != pred.shape:
            self.exp.log_text('error', 'Error: The shape of the target and prediction for the loss calculation is different')
            self.exp.log_text('error', f'{target.shape} {pred.shape}')
            sendToDiscord('There was an error with a loss calcuation, please check training logs')
            return torch.zeros(1).to(self.params.device)
        sum = torch.sum(((target - pred) ** 2) * aligned_status) / data_size
        if math.isnan(sum):
            print('Loss function is nan')
            # print(target)
            # print(pred)
            # print(aligned_status)
            # print(data_size)
        return sum

    def count_paramters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def zero_grad_optim(self):
        pass

    def init_optimizer(self, model):
        pass

    def step_optimizer(self, model, total_loss):
        pass

    def save_params(self, folder, exp:Experiment):
        pass
