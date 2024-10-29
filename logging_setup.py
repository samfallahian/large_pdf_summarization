import logging
import os
from pathlib import Path
from constants import LOG_FILE

def initialize_log_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    with open(file_path, 'w') as file:
        file.write('')  # Create an empty log file
        print(f"{file_path} has been created.")

class CustomFormatter(logging.Formatter):
    # Custom color formatting for logs
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(log_file=LOG_FILE):
    initialize_log_file(log_file)
    formatter = CustomFormatter()
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger = logging.getLogger("gradio_log")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(handler)
    return logger