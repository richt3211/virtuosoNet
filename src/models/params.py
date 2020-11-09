from dataclasses import dataclass, asdict, field
import logging
import json
@dataclass
class Params():
  def __post_init__(self):
    logging.info(json.dumps(asdict(self), indent=4))