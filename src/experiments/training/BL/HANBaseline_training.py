

from datetime import datetime
from src.experiments.training.model_run_job import ModelRun
import src.old.data_process as dp
import math 
import random 
import torch 
import copy

class HANBaselineModelRun(ModelRun):

    def __init__(self, device):
        super().__init__(device)
        self.num_input = 78
        self.num_output = 11
        self.num_prime_param = 11
        
        self.num_updated = 0
        self.time_steps = 500
        self.batch_size = 1

        self.num_key_augmentation = 1

        self.kld_sig = 20e4
        self.kld_max = 0.01

        self.learning_rate = 0.003
        self.weight_decay = 1e-5
        self.grad_clip = 5

        self.feature_loss_init = {
                'tempo': {
                    'name': 'Tempo',
                    'values': []
                },
                'vel': {
                    'name': 'Velocity',
                    'values': []
                },
                'dev': {
                    'name': 'Deviation',
                    'values': []
                },
                'articul': {
                    'name': 'Articulation',
                    'values': []
                },
                'pedal': {
                    'name': 'Pedal',
                    'values': []
                },
                'trill': {
                    'name': 'Trill',
                    'values': []
                },
                'kld': {
                    'name': 'KLD',
                    'values': []
                }
            }


    def sigmoid(x, gain=1):
        return 1 / (1 + math.exp(-gain*x))

    def train(self, model, data, num_epochs):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        now = datetime.now()
        current_time = now.strftime("%D %I:%M:%S")
        print(f'Starting training job at {current_time}')


        train_xy = data['train']
        valid_xy = data['valid']
        for epoch in range(num_epochs):
            print(f'Training Epoch {epoch + 1}')
            # print('current training step is ', NUM_UPDATED)
            

            feature_loss = copy.deepcopy(self.feature_loss_init)
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
                    kld_weight = self.sigmoid((self.num_updated - self.kld_sig) / (self.kld_sig/10)) * self.kld_max

                    for slice_idx in slice_indexes:
                        training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                         'note_locations': note_locations,
                                         'align_matched': align_matched, 'pedal_status': pedal_status,
                                         'slice_idx': slice_idx, 'kld_weight': kld_weight}

                        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                            self.batch_time_step_run(training_data, model=model)
                        feature_loss['tempo'].append(tempo_loss.item())
                        feature_loss['vel'].append(vel_loss.item())
                        feature_loss['dev'].append(dev_loss.item())
                        feature_loss['articul'].append(articul_loss.item())
                        feature_loss['pedal'].append(pedal_loss.item())
                        feature_loss['trill'].append(trill_loss.item())
                        feature_loss['kld'].append(kld.item())
                        self.num_updated += 1

            print('Training Loss')
            self.print_loss(feature_loss)

                    ## Validation
            validation_feature_loss = copy.deepcopy(self.feature_loss_init)

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
                batch_y = batch_y.view(1, -1, self.num_outpute)
                # input_y = torch.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(DEVICE)
                align_matched = torch.Tensor(align_matched).view(1, -1, 1).to(self.device)
                pedal_status = torch.Tensor(pedal_status).view(1,-1,1).to(self.device)
                outputs, total_z = run_model_in_steps(batch_x, batch_y, graphs, note_locations)

                # valid_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM:-num_trill_param], batch_y[:,:,NUM_TEMPO_PARAM:-num_trill_param], align_matched)

                tempo_loss = cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
                vel_loss = criterion(outputs[:, :, VEL_PARAM_IDX], batch_y[:, :, VEL_PARAM_IDX], align_matched)
                deviation_loss = criterion(outputs[:, :, DEV_PARAM_IDX], batch_y[:, :, DEV_PARAM_IDX], align_matched)
                articul_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX], batch_y[:, :, PEDAL_PARAM_IDX], pedal_status)
                pedal_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX+1:], batch_y[:, :, PEDAL_PARAM_IDX+1:], align_matched)
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

            mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss + mean_articul_loss + mean_pedal_loss * 7 + mean_kld_loss * kld_weight) / (11 + kld_weight)

            print("Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
                .format(mean_valid_loss, mean_tempo_loss , mean_vel_loss,
                        mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss))

            is_best = mean_valid_loss < best_prime_loss
            best_prime_loss = min(mean_valid_loss, best_prime_loss)

            is_best_trill = mean_trill_loss < best_trill_loss
            best_trill_loss = min(mean_trill_loss, best_trill_loss)

            if args.trainTrill:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': MODEL.state_dict(),
                    'best_valid_loss': best_trill_loss,
                    'optimizer': optimizer.state_dict(),
                    'training_step': NUM_UPDATED
                }, is_best_trill, model_name='trill')
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': MODEL.state_dict(),
                    'best_valid_loss': best_prime_loss,
                    'optimizer': optimizer.state_dict(),
                    'training_step': NUM_UPDATED
                }, is_best, model_name='prime')


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
            = model_train(prime_batch_x, prime_batch_y, edges, data['note_locations'], batch_start)


        tempo_loss = self.criterion(outputs[:, :, 0:1],
                                prime_batch_y[:, :, 0:1], align_matched)
        vel_loss = self.criterion(outputs[:, :, self.vel_param_idx:self.dev_param_idx],
                            prime_batch_y[:, :, self.vel_param_idx:self.dev_param_idx], align_matched)
        dev_loss = self.criterion(outputs[:, :, self.dev_param_idx:self.articul_param_idx],
                            prime_batch_y[:, :, self.dev_param_idx:self.articul_param_idx], align_matched)
        articul_loss = self.criterion(outputs[:, :, self.articul_param_idx:self.pedal_param_idx],
                                prime_batch_y[:, :, self.articul_param_idx:self.pedal_param_idx], pedal_status)
        pedal_loss = self.criterion(outputs[:, :, self.pedal_param_idx:], prime_batch_y[:, :, self.pedal_param_idx:],
                            align_matched)
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

        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), perform_kld

        # loss = criterion(outputs, batch_y)
        # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])

    def run_model_in_steps(self, input, input_y, edges, note_locations, model, initial_z=False):
        num_notes = input.shape[1]
        with torch.no_grad():  # no need to track history in validation
            model_eval = model.eval()
            total_output = []
            total_z = []
            measure_numbers = [x.measure for x in note_locations]
            slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=VALID_STEPS, overlap=False)
            # if edges is not None:
            #     edges = edges.to(DEVICE)

            for slice_idx in slice_indexes:
                batch_start, batch_end = slice_idx
                if edges is not None:
                    batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end].to(DEVICE)
                else:
                    batch_graph = None
                
                batch_input = input[:, batch_start:batch_end, :].view(1,-1,model.input_size)
                batch_input_y = input_y[:, batch_start:batch_end, :].view(1,-1,model.output_size)
                temp_outputs, perf_mu, perf_var, _ = model_eval(batch_input, batch_input_y, batch_graph,
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
            print('Error: The shape of the target and prediction for the loss calculation is different')
            print(target.shape, pred.shape)
            return torch.zeros(1).to(DEVICE)
        return torch.sum(((target - pred) ** 2) * aligned_status) / data_size