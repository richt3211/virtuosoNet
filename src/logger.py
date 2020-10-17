import logging
import os
import sys

def init_logger(filename):
    if (os.path.exists(filename)):
        os.remove(filename)
    logger = logging.getLogger()
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")

    # set log to file 
    fhandler = logging.FileHandler(filename=filename, mode='w')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)

    # set log to stdout
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.INFO)
