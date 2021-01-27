import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 
import json 
import numpy as np

from dataclasses import dataclass, asdict, field

from torch.nn.modules.normalization import LayerNorm
from src.models.params import Params

@dataclass
class TransformerHyperParams(Params):
    num_encoder_head:int = 6
    num_decoder_head:int = 11
    hidden_size:int = 256
    num_encoder_layers:int = 6
    num_decoder_layers:int = 6
    dropout:float = 0.1

@dataclass
class TransformerEncoderHyperParams(Params):
    with_layer_norm:bool = True
    num_head:int = 6
    hidden_size:int = 256
    num_layers:int = 6
    dropout:float = 0.1

class TransformerEncoder(nn.Module):

    def __init__(self, params:TransformerEncoderHyperParams):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.params = params
        self.model_type = 'Transformer Encoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(params.input_size, params.dropout)
        encoder_layers = TransformerEncoderLayer(params.input_size, params.num_head, params.hidden_size, params.dropout)

        if params.with_layer_norm:
            norm = LayerNorm(params.input_size)    
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, params.num_layers, norm)
        else:
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, params.num_layers)
        self.input_size = params.input_size
        self.decoder = nn.Linear(params.input_size, params.output_size)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # https://stackoverflow.com/a/55546528
        # self.transformer_encoder.weight.data.normal_(mean=0.0, std=1/np.sqrt(self.input_size))
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device) 
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class Transformer(nn.Module):
    def __init__(self, params:TransformerHyperParams):
        super().__init__()
        self.params = params
        self.model_type = 'Transformer'
        self.tgt_mask = None

        self.input_pos_encoder = PositionalEncoding(params.input_size, params.dropout)
        self.output_pos_encoder = PositionalEncoding(params.output_size, params.dropout)

        # using customer encoder and decoder because of a mismatch in the input size 
        encoder_layers = nn.TransformerEncoderLayer(params.input_size, params.num_encoder_head, params.hidden_size, params.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, params.num_encoder_layers)

        decoder_layers = nn.TransformerDecoderLayer(params.output_size, params.num_decoder_head, params.hidden_size, params.dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, params.num_decoder_layers)

        # Passing in first parameters to avoid errors. All parameters should not matter except for the last two. See PyTorch Transformer source. 
        # https://pytorch.org/docs/master/_modules/torch/nn/modules/transformer.html#Transformer
        self.transformer = nn.Transformer(
            d_model=params.input_size, 
            nhead=params.num_encoder_head,
            num_encoder_layers=params.num_encoder_layers,
            num_decoder_layers=params.num_decoder_layers,
            dim_feedforward=params.hidden_size,
            dropout=params.dropout,
            custom_encoder=self.transformer_encoder,
            custom_decoder=self.transformer_decoder
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        if self.tgt_mask is None or self.tgt_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device) 
            self.tgt_mask = mask

        src = self.input_pos_encoder(src)
        tgt = self.output_pos_encoder(tgt)
        output = self.transformer(src, tgt, tgt_mask=self.tgt_mask)
        return output    


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 !=0:
            div_term = div_term[:-1]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)