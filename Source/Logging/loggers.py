from Utils.paths import *
import logging

loggers = {}


def get_logger(name, fname=None):
    global loggers

    if fname in loggers:
        return loggers[fname]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if fname:
        file_handler = logging.FileHandler(os.path.join(log_files_folder, f"{fname}"), encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    loggers[fname] = logger

    return logger
