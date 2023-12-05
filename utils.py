"""
Constants and functions for PauliComposer and PauliDecomposer classes.

See: https://arxiv.org/abs/2301.00560
"""

import numpy as np
from numbers import Real


# Definition of some useful constants
PAULI_LABELS = ['I', 'X', 'Y', 'Z']
NUM2LABEL = {ind: PAULI_LABELS[ind] for ind in range(len(PAULI_LABELS))}
BINARY = {'I': '0', 'X': '1', 'Y': '1', 'Z': '0'}
PAULI = {'I': np.eye(2, dtype=np.uint8),
         'X': np.array([[0, 1], [1, 0]], dtype=np.uint8),
         'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
         'Z': np.array([[1, 0], [0, -1]], dtype=np.int8)}


def nbytes(size: int, n_items: int) -> Real:
    """Return number of bytes needed for a `n_items`-array of `size` bits."""
    # Bits/element * number of elements / 8 bits/byte
    n_bytes = size * n_items / 8
    return int(n_bytes) if n_bytes.is_integer() else n_bytes


def convert_bytes(n_bytes: Real) -> str:
    """Convert a number of bytes `n_bytes` into a manipulable quantity."""
    for unit in ['iB', 'kiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB']:
        if n_bytes < 1024:
            return '%4.2f %s' % (n_bytes, unit)
        n_bytes /= 1024
    return '%4.2f YiB' % (n_bytes)
