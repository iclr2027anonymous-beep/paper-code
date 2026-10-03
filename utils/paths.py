"""Resolve bundled resources independently of the invoking directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(relative: str = "") -> str:
    """Return an absolute path inside this distribution (no private fallback)."""
    return str(PROJECT_ROOT / relative)
