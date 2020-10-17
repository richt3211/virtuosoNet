import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import logging

VEL_PARAM_IDX = 1
DEV_PARAM_IDX = 2
ARTICUL_LOSS_IDX = 3
PEDAL_PARAM_IDX = 4


def train_model(data, model, epochs):
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    feature_loss = {
        'tempo': [],
        'vel': [],
        'dev': [],
        'articul': [],
        'pedal': []
    }
    total_loss = []

    validation_feature_loss = {
        'tempo': [],
        'vel': [],
        'dev': [],
        'articul': [],
        'pedal': []
    }
    validation_total_loss = []
    for epoch in range(epochs):
        logging.info(f'Training for epoch {epoch+1}/{epochs}')
        logging.info("")
        for xy_tuple in data['train']:
            x = xy_tuple[0]
            y = xy_tuple[1]
            # align_matched = xy_tuple[3]
            # pedal_status = xy_tuple[4]
            # edges = xy_tuple[5]

            pred = model_inference(model, x) 
            target = transform_target(y)

            loss = calculate_loss(pred, target, total_loss, feature_loss, loss_function)
            loss.backward()
            optimizer.step()

        logging.info("Training Loss")
        print_loss(feature_loss, total_loss)


        # output the validation loss
        with torch.no_grad():
            for xy_tuple in data['valid']:
                x = xy_tuple[0]
                y = xy_tuple[1]
                # align_matched = xy_tuple[3]
                # pedal_status = xy_tuple[4]
                # edges = xy_tuple[5]

                pred = model_inference(model, x) 
                target = transform_target(y)
                calculate_loss(pred, target, validation_total_loss, validation_feature_loss, loss_function)

        logging.info("Validation Loss and Evaluation")
        print_loss(validation_feature_loss, validation_total_loss)
        logging.info("")
    return model

def transform_target(y):
    target = torch.tensor(y, dtype=torch.float)
    target = target.view(len(y), 1, -1)
    target = target[:, :, 0:11]
    return target 

def model_inference(model, x):
    input = torch.tensor(x, dtype=torch.float)
    input = input.view(len(x), 1, -1)
    return model(input)

def calculate_loss(pred, target, total_loss, feature_loss, loss_function):
    tempo = loss_function(pred[:, :, 0:1], target[:,:, 0:1]).item()
    feature_loss['tempo'].append(tempo)

    vel = loss_function(pred[:, :, VEL_PARAM_IDX: DEV_PARAM_IDX], target[:,:, VEL_PARAM_IDX:DEV_PARAM_IDX]).item()
    feature_loss['vel'].append(vel)

    dev = loss_function(pred[:, :, DEV_PARAM_IDX: PEDAL_PARAM_IDX], target[:,:, DEV_PARAM_IDX:PEDAL_PARAM_IDX]).item()
    feature_loss['dev'].append(dev)

    articul = loss_function(pred[:, :, ARTICUL_LOSS_IDX:PEDAL_PARAM_IDX ], target[:,:, ARTICUL_LOSS_IDX:PEDAL_PARAM_IDX]).item()
    feature_loss['articul'].append(articul)

    pedal = loss_function(pred[:, :, PEDAL_PARAM_IDX:], target[:,:, PEDAL_PARAM_IDX:]).item()
    feature_loss['pedal'].append(pedal)

    loss = loss_function(pred, target)
    total_loss.append(loss.item())
    return loss

def print_loss(feature_loss, loss):
    logging.info(f'Total Loss: {np.mean(loss)}')
    loss_string = "\t"
    for key, value in feature_loss.items():
        loss_string += f'{key}: {np.mean(value):.4} '
    logging.info(loss_string)
    logging.info("")