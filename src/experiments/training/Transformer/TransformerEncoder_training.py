from src.experiments.training.model_run_job import ModelRun
from src.discord_bot import sendToDiscord

import torch
import logging
import copy
import src.old.data_process as dp
import math 
import random 
import torch 
import copy
import numpy as np


class TransformerEncoderTraining(ModelRun):
    def __init__(self, device, is_dev):
        super().__init__(device, is_dev)

        self.learning_rate = 0.5
        self.num_key_augmentation = 1
        self.grad_clip = 0.5

    def init_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-09) 
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def get_batch_and_alignment(self, data):
        batch_start, batch_end = data['slice_idx']
        batch_x, batch_y = self.handle_data_in_tensor(data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

        batch_x = batch_x.view((self.batch_size, -1, self.num_input))
        batch_y = batch_y.view((self.batch_size, -1, self.num_output))

        align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end]).view((self.batch_size, -1, 1)).to(self.device)
        pedal_status = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view((self.batch_size, -1, 1)).to(self.device)

        edges = data['graphs']

        prime_batch_x = batch_x
        prime_batch_y = batch_y[:, :, 0:self.num_prime_param]

        return {
            'batch_x': prime_batch_x,
            'batch_y': prime_batch_y,
            'align_matched': align_matched,
            'pedal_status': pedal_status
        }

    def step_optimizers(self, model, total_loss):
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self.optimizer.step()

    def batch_time_step_run(self, data, model):
        batches = self.get_batch_and_alignment(data)

        model_train = model.train()
        batch_x = batches['batch_x']
        outputs = model_train(batch_x)

        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, total_loss = self.calculate_loss(outputs, batches)
        self.step_optimizers(model, total_loss)

        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), total_loss

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

    def evaluate(self, model, valid_data, epoch, version):
