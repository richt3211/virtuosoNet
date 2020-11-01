

from datetime import datetime
# from src.experiments.models.model_run_job import ModelJob
from src.models.BL import HANBaseline, HANBaselineHyperParams
from src.discord_bot import sendToDiscord
from src.models.model_run_job import ModelJob

import src.old.data_process as dp
import math 
import random 
import torch 
import copy
import numpy as np
import logging

class HANBaselineModelRun():

    def __init__(self, device, is_dev):
        self.num_input = 78
        self.num_output = 11
        self.num_prime_param = 11
        self.num_tempo_param = 1

        self.qpm_index = 0
        self.valid_steps = 5000

        self.num_updated = 0
        self.time_steps = 500
        self.batch_size = 1

        self.num_key_augmentation = 1

        self.kld_sig = 20e4
        self.kld_max = 0.01
        self.kld_weight = 0

        self.learning_rate = 0.003
        self.weight_decay = 1e-5
        self.grad_clip = 5

        self.device = device 
        self.is_dev = is_dev

        self.qpm_index = 0
        self.vel_param_idx = 1
        self.dev_param_idx = 2
        self.articul_param_idx = 3
        self.pedal_param_idx = 4

        self.feature_loss_init = {
            'tempo': [],
            'vel': [],
            'dev': [],
            'articul': [],
            'pedal': [],
            'trill': [],
            'kld': []
        }



    def sigmoid(self, x, gain=1):
        return 1 / (1 + math.exp(-gain*x))

    def train(self, model, data, num_epochs, version):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        now = datetime.now()
        current_time = now.strftime("%D %I:%M:%S")
        logging.info(f'Starting training job at {current_time}')

        best_loss = float("inf")

        train_xy = data['train']
        valid_xy = data['valid']
        for epoch in range(num_epochs):
            logging.info(f'Training Epoch {epoch + 1}')
            logging.info("")

            feature_loss = copy.deepcopy(self.feature_loss_init)
            total_loss = []
            for xy_tuple in train_xy:
                train_x = xy_tuple[0]
                train_y = xy_tuple[1]
                note_locations = xy_tuple[2]
                align_matched = xy_tuple[3]
                pedal_status = xy_tuple[4]
                edges = xy_tuple[5]

                data_size = len(note_locations)

                # keeping this here from old training loop
                graphs = None

                measure_numbers = [x.measure for x in note_locations]
                # graphs = edges_to_sparse_tensor(edges)
                total_batch_num = int(math.ceil(data_size / (self.time_steps * self.batch_size)))

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
                        training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
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

            logging.info('Training Loss')
            self.print_loss(feature_loss, total_loss)

            validation_feature_loss = copy.deepcopy(self.feature_loss_init)
            validation_loss = []
            for xy_tuple in valid_xy:
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

            logging.info('Validation loss')
            self.print_loss(validation_feature_loss, validation_loss)
            logging.info("")

            mean_valid_loss = np.mean(validation_loss)
            if ((epoch+1) >= 5 and (epoch+1) % 5 == 0):
                sendToDiscord(f'Trained model for {epoch + 1} epochs with loss of {mean_valid_loss}')

            is_best = mean_valid_loss < best_loss
            best_loss = min(mean_valid_loss, best_loss)

            # self.save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'best_valid_loss': best_loss,
            #     'optimizer': self.optimizer.state_dict(),
            #     'training_step': self.num_updated
            # }, is_best, "BL/HAN_BL", version)

    def batch_time_step_run(self, data, model):
        batch_start, batch_end = data['slice_idx']
        batch_x, batch_y = self.handle_data_in_tensor(data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

        batch_x = batch_x.view((self.batch_size, -1, self.num_input))
        batch_y = batch_y.view((self.batch_size, -1, self.num_output))

        align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end]).view((self.batch_size, -1, 1)).to(self.device)
        pedal_status = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view((self.batch_size, -1, 1)).to(self.device)

        edges = data['graphs']

        prime_batch_x = batch_x
        prime_batch_y = batch_y[:, :, 0:self.num_prime_param]

        model_train = model.train()
        outputs, perform_mu, perform_var, total_out_list \
            = model_train(prime_batch_x, prime_batch_y, data['note_locations'], batch_start)


        tempo_loss = self.criterion(outputs[:, :, 0:1], prime_batch_y[:, :, 0:1], align_matched)
        vel_loss = self.criterion(outputs[:, :, self.vel_param_idx:self.dev_param_idx], prime_batch_y[:, :, self.vel_param_idx:self.dev_param_idx], align_matched)
        dev_loss = self.criterion(outputs[:, :, self.dev_param_idx:self.articul_param_idx], prime_batch_y[:, :, self.dev_param_idx:self.articul_param_idx], align_matched)
        articul_loss = self.criterion(outputs[:, :, self.articul_param_idx:self.pedal_param_idx], prime_batch_y[:, :, self.articul_param_idx:self.pedal_param_idx], pedal_status)
        pedal_loss = self.criterion(outputs[:, :, self.pedal_param_idx:], prime_batch_y[:, :, self.pedal_param_idx:], align_matched)
        total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11

        if isinstance(perform_mu, bool):
            perform_kld = torch.zeros(1)
        else:
            perform_kld = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
            total_loss += perform_kld * data['kld_weight']
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self.optimizer.step()

        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), perform_kld, total_loss

        # loss = criterion(outputs, batch_y)
        # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])

    def run_model_in_steps(self, input, input_y, edges, note_locations, model: HANBaseline, initial_z=False):
        num_notes = input.shape[1]
        with torch.no_grad():  # no need to track history in validation
            model_eval = model.eval()
            total_output = []
            total_z = []
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=self.valid_steps, overlap=False)
            # if edges is not None:
            #     edges = edges.to(DEVICE)
            for slice_idx in slice_indexes:
                batch_start, batch_end = slice_idx
                if edges is not None:
                    batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end].to(self.device)
                else:
                    batch_graph = None
                
                batch_input = input[:, batch_start:batch_end, :].view(1,-1,model.input_size)
                batch_input_y = input_y[:, batch_start:batch_end, :].view(1,-1,model.output_size)
                temp_outputs, perf_mu, perf_var, _ = model_eval(batch_input, batch_input_y,
                                                                            note_locations=note_locations, start_index=batch_start, initial_z=initial_z)
                total_z.append((perf_mu, perf_var))
                total_output.append(temp_outputs)

            outputs = torch.cat(total_output, 1)
            return outputs, total_z

    def handle_data_in_tensor(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        y = y[:, :self.num_prime_param]

        return x.to(self.device), y.to(self.device)

    def criterion(self, pred, target, aligned_status=1):
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

    def cal_tempo_loss_in_beat(self, pred_x, true_x, note_locations, start_index):
        previous_beat = -1

        num_notes = pred_x.shape[1]
        start_beat = note_locations[start_index].beat
        num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1

        pred_beat_tempo = torch.zeros([num_beats, self.num_tempo_param]).to(self.device)
        true_beat_tempo = torch.zeros([num_beats, self.num_tempo_param]).to(self.device)
        for i in range(num_notes):
            current_beat = note_locations[i+start_index].beat
            if current_beat > previous_beat:
                previous_beat = current_beat
                for j in range(i, num_notes):
                    if note_locations[j+start_index].beat > current_beat:
                        break
                if not i == j:
                    pred_beat_tempo[current_beat - start_beat] = torch.mean(pred_x[0, i:j, self.qpm_index])
                    true_beat_tempo[current_beat - start_beat] = torch.mean(true_x[0, i:j, self.qpm_index])

        tempo_loss = self.criterion(pred_beat_tempo, true_beat_tempo)
        # if args.deltaLoss and pred_beat_tempo.shape[0] > 1:
        #     prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
        #     true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
        #     delta_loss = criterion(prediction_delta, true_delta)

        #     tempo_loss = (tempo_loss + delta_loss * DELTA_WEIGHT) / (1 + DELTA_WEIGHT)

        return tempo_loss

    def print_loss(self, feature_loss, loss):
        logging.info(f'Total Loss: {np.mean(loss)}')
        loss_string = "\t"
        for key, value in feature_loss.items():
            loss_string += f'{key}: {np.mean(value):.4} '
        logging.info(loss_string)
        logging.info("")

def run_han_bl_job(data, num_epochs, version, is_dev):
    try:
        type = "DEV" if is_dev else ""
        start_message = f"STARTING HAN BL TRAINING VERSION {version} JOB AT {num_epochs} EPOCHS FOR {type} DATA SET"
        logging.info(start_message)
        sendToDiscord(start_message)
        device = 1

        hyper_params = HANBaselineHyperParams()
        model = HANBaseline(hyper_params, device).to(device)
        
        job = HANBaselineModelRun(device, is_dev)
        job.train(model, data, num_epochs, version)
        end_message = f'FINISHED HAN BL VERSION {version} TRAINING JOB AT {num_epochs} EPOCHS FOR {type} DATA SET'
        logging.info(end_message)
        sendToDiscord(end_message)
    except Exception as e:
        logging.exception("Error during training")
        sendToDiscord("There was an error during training for the HAN BL training job, please check logs")
        raise e