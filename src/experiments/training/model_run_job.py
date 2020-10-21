from src.logger import init_logger
from src.constants import CACHE_MODEL_DIR
from src.discord_bot import sendToDiscord
import src.old.data_process as dp

import torch 
import numpy as np
import logging
import os
import shutil
import torch
import math 
import random 
import copy



logger = logging.getLogger()
class ModelRun():

    def __init__(self, device, is_dev):
        torch.cuda.set_device(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vel_param_idx = 1
        self.dev_param_idx = 2
        self.articul_param_idx = 3
        self.pedal_param_idx = 4

        self.is_dev = is_dev
        self.feature_loss_init = {
            'tempo': [],
            'vel': [],
            'dev': [],
            'articul': [],
            'pedal': [],
            'trill': [],
            'kld': []
        }

    def train(self, model, data, num_epochs, version):
        best_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_num = epoch +1
            logging.info(f'Training Epoch {epoch_num}')
            logging.info("")

            total_loss, feature_loss = self.train_epoch(model, data['train'], epoch_num)
            logging.info('Training Loss')
            self.print_loss(feature_loss, total_loss)

            total_valid_loss, valid_feature_loss = self.evaluate(model, data['valid'])
            logging.info('Validation loss')
            self.print_loss(total_valid_loss, valid_feature_loss)
            logging.info("")

            mean_valid_loss = np.mean(total_valid_loss)
            # if ((epoch_num) >= 5 and (epoch_num) % 5 == 0):
            sendToDiscord(f'Trained model for {epoch_num} epochs with validation loss of {mean_valid_loss}')

            is_best = mean_valid_loss < best_loss
            best_loss = min(mean_valid_loss, best_loss)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_valid_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.num_updated
            }, is_best, "Transformer/TransformerEncoder", version)


    def train_epoch(self, model, train_data):
        self.init_optimizer()
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
            for i in range(self.num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(self.num_key_augmentation+1):
                key = key_lists[i]
                temp_train_x = dp.key_augmentation(train_x, key)
                slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.time_steps)
                self.kld_weight = self.sigmoid((self.num_updated - self.kld_sig) / (self.kld_sig/10)) * self.kld_max

                for slice_idx in slice_indexes:
                    training_data = {'x': temp_train_x, 'y': train_y,
                                        'note_locations': note_locations,
                                        'align_matched': align_matched, 'pedal_status': pedal_status,
                                        'slice_idx': slice_idx, 'kld_weight': self.kld_weight}

                    tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld, loss = \
                        self.batch_time_step_run(training_data, model=model)
                    feature_loss['tempo'].append(tempo_loss.item())
                    feature_loss['vel'].append(vel_loss.item())
                    feature_loss['dev'].append(dev_loss.item())
                    feature_loss['articul'].append(articul_loss.item())
                    feature_loss['pedal'].append(pedal_loss.item())
                    feature_loss['trill'].append(trill_loss.item())
                    feature_loss['kld'].append(kld.item())

                    total_loss.append(loss.item())
                    self.num_updated += 1

        return total_loss, feature_loss

    def evaluate(self, model, valid_data):
        validation_feature_loss = copy.deepcopy(self.feature_loss_init)
        validation_loss = []
        for xy_tuple in valid_data:
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]
            edges = xy_tuple[5]
            graphs = None


            batch_x, batch_y = self.handle_data_in_tensor(test_x, test_y)
            batch_x = batch_x.view(1, -1, self.num_input)
            batch_y = batch_y.view(1, -1, self.num_output)
            # input_y = torch.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(DEVICE)
            align_matched = torch.Tensor(align_matched).view(1, -1, 1).to(self.device)
            pedal_status = torch.Tensor(pedal_status).view(1,-1,1).to(self.device)
            outputs, total_z = self.run_model_in_steps(batch_x, batch_y, graphs, note_locations, model)

            tempo_loss = self.cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
            vel_loss = self.criterion(outputs[:, :, self.vel_param_idx], batch_y[:, :, self.vel_param_idx], align_matched)
            deviation_loss = self.criterion(outputs[:, :, self.dev_param_idx], batch_y[:, :, self.dev_param_idx], align_matched)
            articul_loss = self.criterion(outputs[:, :, self.articul_param_idx], batch_y[:, :, self.articul_param_idx], pedal_status)
            pedal_loss = self.criterion(outputs[:, :, self.pedal_param_idx:], batch_y[:, :, self.pedal_param_idx:], align_matched)
            trill_loss = torch.zeros(1)
            for z in total_z:
                perform_mu, perform_var = z
                kld_loss = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                validation_feature_loss['kld'].append(kld_loss.item())

            validation_feature_loss['tempo'].append(tempo_loss.item())
            validation_feature_loss['vel'].append(vel_loss.item())
            validation_feature_loss['dev'].append(deviation_loss.item())
            validation_feature_loss['articul'].append(articul_loss.item())
            validation_feature_loss['pedal'].append(pedal_loss.item())
            validation_feature_loss['trill'].append(trill_loss.item())

            loss = (tempo_loss + vel_loss + deviation_loss + articul_loss + pedal_loss * 7 + kld_loss * self.kld_weight) / (11 + self.kld_weight)

            validation_loss.append(loss.item())

    def init_optimizer(self):
        pass

    def batch_time_step_run(self, data, model):
        pass

    def calculate_loss(self, outputs, batches):
        prime_batch_y = batches['batch_y']
        align_matched = batches['align_matched']
        pedal_status = batches['pedal_status']

        tempo_loss = self.han_criterion(outputs[:, :, 0:1], prime_batch_y[:, :, 0:1], align_matched)
        vel_loss = self.han_criterion(outputs[:, :, self.vel_param_idx:self.dev_param_idx], prime_batch_y[:, :, self.vel_param_idx:self.dev_param_idx], align_matched)
        dev_loss = self.han_criterion(outputs[:, :, self.dev_param_idx:self.articul_param_idx], prime_batch_y[:, :, self.dev_param_idx:self.articul_param_idx], align_matched)
        articul_loss = self.han_criterion(outputs[:, :, self.articul_param_idx:self.pedal_param_idx], prime_batch_y[:, :, self.articul_param_idx:self.pedal_param_idx], pedal_status)
        pedal_loss = self.han_criterion(outputs[:, :, self.pedal_param_idx:], prime_batch_y[:, :, self.pedal_param_idx:], align_matched)
        total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11

        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_status, torch.zeros(1), total_loss

    def handle_data_in_tensor(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        y = y[:, :self.num_prime_param]

        return x.to(self.device), y.to(self.device)

    def print_loss(self, feature_loss, loss):
        logging.info(f'Total Loss: {np.mean(loss)}')
        loss_string = "\t"
        for key, value in feature_loss.items():
            loss_string += f'{key}: {np.mean(value):.4} '
        logging.info(loss_string)
        logging.info("")

    def save_checkpoint(self, state, is_best, folder, version):
        folder = f'{CACHE_MODEL_DIR}/{folder}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        if self.is_dev:
            filepath = f'{folder}/v{version}_dev.pth'
        else:
            filepath = f'{folder}/v{version}.pth'
        torch.save(state, filepath)
        if is_best:
            if self.is_dev:
                best_filepath = f'{folder}/v{version}_dev_best.pth'
            else:
                best_filepath = f'{folder}/v{version}_best.pth'
            shutil.copyfile(filepath, best_filepath)

    def han_criterion(self, pred, target, aligned_status=1):
        if isinstance(aligned_status, int):
            data_size = pred.shape[-2] * pred.shape[-1]
        else:
            data_size = torch.sum(aligned_status).item() * pred.shape[-1]
            if data_size == 0:
                data_size = 1
        if target.shape != pred.shape:
            logging.error('Error: The shape of the target and prediction for the loss calculation is different')
            logging.error(target.shape, pred.shape)
            sendToDiscord('There was an error with a loss calcuation, please check training logs')
            return torch.zeros(1).to(self.device)
        return torch.sum(((target - pred) ** 2) * aligned_status) / data_size