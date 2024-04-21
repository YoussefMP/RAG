from Source.paths import *
import logging


def get_logger(name, fname=None):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if fname:
        file_handler = logging.FileHandler(os.path.join(log_files_folder, f"{fname}"), encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
