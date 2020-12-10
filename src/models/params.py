from dataclasses import dataclass, asdict, field
import logging
import json

import torch
from torch import device
@dataclass
class Params():
  input_size:int = 78
  output_size:int = 11
  device_num:int = 1 
  time_steps:int = 500
  is_dev:bool = False
  device:device = torch.device(f'cuda:{device_num}')
