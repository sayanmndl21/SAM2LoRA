import logging
import os
from typing import Optional

def setup_logger(log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("sam2lora_logger")
    
    # Clear any existing handlers to avoid duplicate logs
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        handlers = [console_handler]
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
            file_handler.setLevel(logging.DEBUG)
            handlers.append(file_handler)
        
        # Set logging levels for handlers
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    return logger
