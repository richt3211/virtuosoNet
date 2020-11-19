from dataclasses import dataclass, asdict, field
import logging
import json
@dataclass
class Params():
  input_size:int = 78
  output_size:int = 11
  def __post_init__(self):
    logging.info(json.dumps(asdict(self), indent=4))