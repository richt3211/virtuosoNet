from dataclasses import dataclass, asdict, field
import logging
import json

import torch
@dataclass
class Params():
  input_size:int = 78
  output_size:int = 11
  device_num:int = 1 
  device = torch.device(f'cuda:{device_num}')
  time_steps = 500
  is_dev:bool = False
