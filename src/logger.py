import logging
import os
import sys

initialized = False

def init_logger(filename):
    global initialized
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
    if initialized == False:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(logging.INFO)
        initialized = True
