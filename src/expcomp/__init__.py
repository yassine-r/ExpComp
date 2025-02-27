from .condition import Condition


import sys
from loguru import logger


def setup_logger():
    # Remove default handler
    logger.remove()

    # Add console handler with INFO level and improved formatting
    logger.add(
        sink=sys.stderr,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    return logger


# Initialize the logger
logger = setup_logger()
