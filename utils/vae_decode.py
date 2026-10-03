"""Unified helper for unpacking a VAE decoder's output."""
from __future__ import annotations


def unbox_decoder_output(out):
    """Return the first element when a decoder returns a tuple/list, else ``out``."""
    return out[0] if isinstance(out, (tuple, list)) else out
