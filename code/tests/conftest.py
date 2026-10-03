"""pytest setup: headless matplotlib backend (no GUI windows during tests)."""
import matplotlib

matplotlib.use("Agg")
