import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except Exception:
    __version__ = "unknown"
