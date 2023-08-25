import datetime
import time

from loguru import logger

SCRIPT_CLOCK_START = time.perf_counter()
START_TIME = datetime.datetime.now()


def fmt_message(message):
    seconds = time.perf_counter() - SCRIPT_CLOCK_START
    fmt_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
    time_string = f"[{fmt_time}] "
    return f"{time_string}{message}"
