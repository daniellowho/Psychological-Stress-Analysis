"""stress_signals: aggregate, population-level stress-related language signals.

Not a diagnostic tool. No individual-level outputs.
"""

from .config import ConfigError, load_config, validate_config
from .utils import detect_environment, get_logger, set_seed, write_manifest

__version__ = "0.0.1"
ETHICS_BANNER = "Aggregate, population-level language signals only. No diagnosis. No individual-level analysis."

__all__ = [
    "ConfigError", "load_config", "validate_config", "detect_environment",
    "get_logger", "set_seed", "write_manifest", "ETHICS_BANNER", "__version__",
]
