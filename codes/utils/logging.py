"""
Logging utilities
"""

import logging
from pathlib import Path


def setup_logging(log_dir: str = "./logs", level: int = logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "transmllm.log"),
            logging.StreamHandler()
        ]
    )

