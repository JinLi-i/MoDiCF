import logging

formatter = logging.Formatter(
    "[ %(levelname)s : %(asctime)s ] - %(message)s")
logging.basicConfig(
    level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
logger = logging.getLogger("Pytorch")

def print(*args, **kwargs):
    logger.info(*args, **kwargs)