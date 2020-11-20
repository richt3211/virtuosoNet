import logging
import os
import sys

from datetime import datetime

from neptune.experiments import Experiment

initialized = False

def init_logger():
    global initialized
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")

    # set log to file 
    # fhandler = logging.FileHandler(filename=filename, mode='w')
    # fhandler.setFormatter(formatter)
    # logger.addHandler(fhandler)

    # set log to stdout
    if initialized == False:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(logging.INFO)
        initialized = True

    return logger


