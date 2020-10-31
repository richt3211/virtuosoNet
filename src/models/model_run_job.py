from src.logger import init_logger
from src.constants import CACHE_MODEL_DIR
from src.discord_bot import sendToDiscord
from src.models.model_writer_reader import save_checkpoint
import src.old.data_process as dp

import torch 
import numpy as np
import logging
import os
import shutil
import torch
import random 
import copy

logger = logging.getLogger()

class ModelJobParams():

    def __init__(self, is_dev):
        self.qpm_index = 0
        self.vel_param_idx = 1
        self.dev_param_idx = 2
        self.articul_param_idx = 3
        self.pedal_param_idx = 4

        self.time_steps = 500
        self.num_key_augmentation = 1
        self.batch_size = 1
        
        self.num_tempo_param = 1
        self.num_input = 78
        self.num_output = 11
        self.num_prime_param = 11
    
        self.device_num=1
        torch.cuda.set_device(self.device_num)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.is_dev = is_dev

class ModelJob():

    def __init__(self, params:ModelJobParams):
        self.params = params
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
        self.model = None
        self.num_updated = 0

    def run_job(self, data, num_epochs, version):
        try:
            type = "DEV" if self.params.is_dev else ""
            start_message = f"STARTING {self.model_name} TRAINING VERSION {version} JOB AT {num_epochs} EPOCHS FOR {type} DATA SET"
            logging.info(start_message)
            sendToDiscord(start_message)

            self.train(self.model, data, num_epochs, version)
            end_message = f'FINISHED {self.model_name} VERSION {version} TRAINING JOB AT {num_epochs} EPOCHS FOR {type} DATA SET'
            logging.info(end_message)
            sendToDiscord(end_message)
        except Exception as e:
            logging.exception("Error during training")
            sendToDiscord("There was an error during training for the HAN BL training job, please check logs")
            raise e

    def train(self, model, data, num_epochs, version):
        best_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_num = epoch +1
            logging.info(f'Training Epoch {epoch_num}')
            logging.info("")

            total_loss, feature_loss = self.train_epoch(model, data['train'])
            logging.info('Training Loss')
            self.print_loss(feature_loss, total_loss)

            total_valid_loss, valid_feature_loss = self.evaluate(model, data['valid'])
            logging.info('Validation loss')
            self.print_loss(valid_feature_loss, total_valid_loss)
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
            self.num_updated += 1

    def init_optimizer(self):
        pass

    def step_optimizer(self, model, total_loss):
        pass

    def calculate_loss(self, outputs, batches):
        prime_batch_y = batches['batch_y']
        align_matched = batches['align_matched']
        pedal_status = batches['pedal_status']

        tempo_loss = self.han_criterion(
            outputs[:, :, 0:1], 
            prime_batch_y[:, :, 0:1], align_matched
        )
        vel_loss = self.han_criterion(
            outputs[:, :, self.params.vel_param_idx:self.params.dev_param_idx], 
            prime_batch_y[:, :, self.params.vel_param_idx:self.params.dev_param_idx], 
            align_matched
        )
        dev_loss = self.han_criterion(
            outputs[:, :, self.params.dev_param_idx:self.params.articul_param_idx], 
            prime_batch_y[:, :, self.params.dev_param_idx:self.params.articul_param_idx], 
            align_matched
        )
        articul_loss = self.han_criterion(
            outputs[:, :, self.params.articul_param_idx:self.params.pedal_param_idx], 
            prime_batch_y[:, :, self.params.articul_param_idx:self.params.pedal_param_idx], 
            pedal_status
        )
        pedal_loss = self.han_criterion(
            outputs[:, :, self.params.pedal_param_idx:], 
            prime_batch_y[:, :, self.params.pedal_param_idx:], 
            align_matched
        )
        total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11

        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_status, torch.zeros(1), total_loss

    def get_batch_and_alignment(self, data):
        batch_start, batch_end = data['slice_idx']
        batch_x, batch_y = self.handle_data_in_tensor(data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

        batch_x = batch_x.view((self.params.batch_size, -1, self.params.num_input))
        batch_y = batch_y.view((self.params.batch_size, -1, self.params.num_output))

        align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end]).view((self.params.batch_size, -1, 1)).to(self.params.
        device)
        pedal_status = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view((self.params.batch_size, -1, 1)).to(self.params.device)

        prime_batch_x = batch_x
        prime_batch_y = batch_y[:, :, 0:self.params.num_prime_param]

        return prime_batch_x, prime_batch_y, align_matched, pedal_status
        # return {
        #     'batch_x': prime_batch_x,
        #     'batch_y': prime_batch_y,
        #     'align_matched': align_matched,
        #     'pedal_status': pedal_status
        # }

    def batch_time_step_run(self, data, model, feature_loss, loss, train=True):
        batch_x, batch_y, align_matched, pedal_status = self.get_batch_and_alignment(data)

        outputs = model(batch_x)

        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, total_loss = self.calculate_loss(outputs, batch_y, align_matched, pedal_status)
        if train:
            self.step_optimizer(model, total_loss)

        feature_loss['tempo'].append(tempo_loss.item())
        feature_loss['vel'].append(vel_loss.item())
        feature_loss['dev'].append(dev_loss.item())
        feature_loss['articul'].append(articul_loss.item())
        feature_loss['pedal'].append(pedal_loss.item())
        feature_loss['trill'].append(trill_loss.item())

        loss.append(total_loss.item())

        return outputs

    def calculate_loss(self, outputs, batch_y, align_matched, pedal_status):
        tempo_loss = self.han_criterion(
            outputs[:, :, 0:1], 
            batch_y[:, :, 0:1], 
            align_matched
        )
        vel_loss = self.han_criterion(
            outputs[:, :, self.params.vel_param_idx:self.params.dev_param_idx], 
            batch_y[:, :, self.params.vel_param_idx:self.params.dev_param_idx], 
            align_matched
        )
        dev_loss = self.han_criterion(
            outputs[:, :, self.params.dev_param_idx:self.params.articul_param_idx], 
            batch_y[:, :, self.params.dev_param_idx:self.params.articul_param_idx], 
            align_matched
        )
        articul_loss = self.han_criterion(
            outputs[:, :, self.params.articul_param_idx:self.params.pedal_param_idx], 
            batch_y[:, :, self.params.articul_param_idx:self.params.pedal_param_idx], 
            pedal_status
        )
        pedal_loss = self.han_criterion(
            outputs[:, :, self.params.pedal_param_idx:], 
            batch_y[:, :, self.params.pedal_param_idx:], 
            align_matched
        )
        total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11

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
        if self.params.is_dev:
            filepath = f'{folder}/v{version}_dev.pth'
        else:
            filepath = f'{folder}/v{version}.pth'
        torch.save(state, filepath)
        if is_best:
            if self.params.is_dev:
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
            return torch.zeros(1).to(self.params.device)
        return torch.sum(((target - pred) ** 2) * aligned_status) / data_size

    