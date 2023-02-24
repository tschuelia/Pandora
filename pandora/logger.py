import logging
import time

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SCRIPT_START = time.perf_counter()


def fmt_message(message):
    seconds = time.perf_counter() - SCRIPT_START
    fmt_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
    time_string = f"[{fmt_time}] "
    return f"{time_string}{message}"
