import argparse
import time 
import torch
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-time", "--timeout", type=int, default=5, help="seconds to timeout")

args = parser.parse_args()

while True:
    logging.info(f'sleeping for {args.timeout} seconds')
    time.sleep(args.timeout)