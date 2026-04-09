import logging
import os
import sys

class LogManager:
    _instances = {}

    def __init__(self, log_dir="./logs"):
        self.log_dir = os.path.abspath(log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_logger(self, name, file_name, level=logging.INFO, to_console=True):
        # 1. Check if we've already configured this specific logger name
        if name in self._instances:
            return self._instances[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 2. Prevent logs from "bubbling up" to the root logger
        # This stops double-printing if the root logger is also configured
        logger.propagate = False 

        # 3. Clear existing handlers (crucial for Notebooks)
        # This ensures that if you re-run the cell, you don't add a 2nd/3rd handler
        if logger.hasHandlers():
            logger.handlers.clear()

        # --- FILE HANDLER (Writes to File) ---
        file_path = os.path.join(self.log_dir, file_name)
        fh = logging.FileHandler(file_path, mode='a') # 'a' for append
        fh.setFormatter(self.formatter)
        logger.addHandler(fh)

        # --- STREAM HANDLER (Prints to Console) ---
        if to_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(self.formatter)
            logger.addHandler(ch)

        self._instances[name] = logger
        return logger

    from contextlib import contextmanager