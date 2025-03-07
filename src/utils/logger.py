"""
Module for logging utilities
"""
import logging

def setup_logger(name, level=logging.INFO):
    """Set up a logger with the specified name and level"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'{name}.log')
    
    # Create formatters and add it to handlers
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    c_format = logging.Formatter(format_str)
    f_format = logging.Formatter(format_str)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger
