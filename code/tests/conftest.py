"""pytest setup: headless matplotlib backend (no GUI windows during tests)."""
try:  # defensive: on one earlier torch build torch had to load before pandas/pyarrow (WinError 1114); harmless otherwise
    import torch  # noqa: F401
except ImportError:
    pass
import matplotlib

matplotlib.use("Agg")
