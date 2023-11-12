from Source.paths import *
import logging


def get_logger(name, fname):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_files_folder, fname))
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
