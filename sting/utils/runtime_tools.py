import logging
import os
import functools
import time
from typing import Callable
import pathlib as Path

def setup_logging_file(case_directory: str):
    """Setup file logging to the specified case directory."""

    file_path = os.path.join(case_directory, "sting_log.txt")
    Path.Path(file_path).touch(exist_ok=True) 
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    #file_handler.terminator = ''  # Remove automatic newline
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Also set terminator for console handler (StreamHandler)
    #for handler in root_logger.handlers:
    #    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
    #        handler.terminator = ''
    
    return file_handler

def timeit(func: Callable):
    """
    A decorator that measures the execution time of the decorated function.
    """
    first_line = func.__doc__.strip().splitlines()[0].strip()
    first_line = first_line.rstrip('.')
    first_line = first_line[0].lower() + first_line[1:]
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"> Initializing {first_line} ... ")
        start_time = time.perf_counter()  # Use perf_counter for more precise timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time)
        logging.info(f"> Completed in {elapsed_time:.2f} seconds. \n")
        return result
    return wrapper