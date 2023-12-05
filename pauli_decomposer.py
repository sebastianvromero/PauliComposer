"""
PauliDecomposer class definition.

See: https://arxiv.org/abs/2301.00560
"""

import warnings
import numpy as np
import itertools as it
from multiprocessing import Array, Manager, Pool

from utils import PAULI_LABELS
from pauli_composer import PauliComposer, PauliDiagComposer


# Ignore ComplexWarning
warnings.simplefilter('ignore', np.ComplexWarning)


# ==============================================================================
# Helper functions for parallelizing weight computation routine
def compute_weights_diag(comb) -> None:
    value = 0
    entry = ''.join(comb)
    pauli_comp = PauliDiagComposer(entry)
    ent = pauli_comp.mat
    for r in rows:
        coef = ent[r]
        ham_term = H_real[r] + 1j*H_imag[r]
        if coef == 1:
            value += ham_term
        elif coef == -1:
            value -= ham_term
        else:
            value += coef * ham_term
    # Store only non-zero values
    if value != 0:
        # Transform non-complex values into float
        if not np.iscomplex(value):
            value = float(value)
        # Divide by 2**n
        coefficients[entry] = value / size
def compute_weights_diag_real(comb) -> None:
    value = 0
    entry = ''.join(comb)
    pauli_comp = PauliDiagComposer(entry)
    ent = pauli_comp.mat
    for r in rows:
        coef = ent[r]
        if coef == 1:
            value += H[r]
        elif coef == -1:
            value -= H[r]
        else:
            value += coef * H[r]
    # Store only non-zero values
    if value != 0:
        # Transform non-complex values into float
        if not np.iscomplex(value):
            value = float(value)
        # Divide by 2**n
        coefficients[entry] = value / size

def compute_weights_general(comb) -> None:
    value = 0
    # Return the Kronecker product of the Pauli matrices and the
    # positions of the non-zero entries
    entry = ''.join(comb)
    if all({comb}) in {'I', 'Z'}:
        pauli_comp = PauliDiagComposer(entry)
        cols, ent = rows, pauli_comp.mat  # NOTE: NEW!
    else:
        pauli_comp = PauliComposer(entry)
        cols, ent = pauli_comp.col, pauli_comp.mat
    for r in rows:
        coef, c = ent[r], cols[r]
        ham_term = H_real[c, r] + 1j*H_imag[c, r]
        if coef == 1:
            value += ham_term
        elif coef == -1:
            value -= ham_term
        else:
            value += coef * ham_term
    # Store only non-zero values
    if value != 0:
        # Transform non-complex values into float
        if not np.iscomplex(value):
            value = float(value)
        # Divide by 2**n
        coefficients[entry] = value / size
def compute_weights_general_real(comb) -> None:
    value = 0
    # Return the Kronecker product of the Pauli matrices and the
    # positions of the non-zero entries
    entry = ''.join(comb)
    if all({comb}) in {'I', 'Z'}:
        pauli_comp = PauliDiagComposer(entry)
        cols, ent = rows, pauli_comp.mat  # NOTE: NEW!
    else:
        pauli_comp = PauliComposer(entry)
        cols, ent = pauli_comp.col, pauli_comp.mat
    for r in rows:
        coef = ent[r]
        if coef == 1:
            value += H[cols[r], r]
        elif coef == -1:
            value -= H[cols[r], r]
        else:
            value += coef * H[cols[r], r]
    # Store only non-zero values
    if value != 0:
        # Transform non-complex values into float
        if not np.iscomplex(value):
            value = float(value)
        # Divide by 2**n
        coefficients[entry] = value / size

def init_pool_diag(int_size, dict_coef, array_rows, array_Hr, array_Hi) -> None:
    global size
    size = int_size
    global coefficients
    coefficients = dict_coef
    global rows
    rows = np.frombuffer(array_rows, dtype='int32')
    global H_real
    H_real = np.frombuffer(array_Hr, dtype='float32')
    global H_imag
    H_imag = np.frombuffer(array_Hi, dtype='float32')
def init_pool_diag_real(int_size, dict_coef, array_rows, array_H) -> None:
    global size
    size = int_size
    global coefficients
    coefficients = dict_coef
    global rows
    rows = np.frombuffer(array_rows, dtype='int32')
    global H
    H = np.frombuffer(array_H, dtype='float32')

def init_pool_general(int_size, dict_coef, array_rows, array_Hr, array_Hi) -> None:
    global size
    size = int_size
    global coefficients
    coefficients = dict_coef
    global rows
    rows = np.frombuffer(array_rows, dtype='int32')
    global H_real
    H_real = np.frombuffer(array_Hr, dtype='float32').reshape(size, size)
    global H_imag
    H_imag = np.frombuffer(array_Hi, dtype='float32').reshape(size, size)
def init_pool_general_real(int_size, dict_coef, array_rows, array_H) -> None:
    global size
    size = int_size
    global coefficients
    coefficients = dict_coef
    global rows
    rows = np.frombuffer(array_rows, dtype='int32')
    global H
    H = np.frombuffer(array_H, dtype='float32').reshape(size, size)
# ==============================================================================

class PauliDecomposer:
    """PauliDecomposer class definition."""

    def __init__(self, H: np.ndarray):
        """Initialize PauliDecomposer class."""
        # Check if all the given hamiltonian elements are real
        self.real_H = np.all(np.isreal(H))
        self.sym = self.real_H and np.all(H == H.T)

        # Number of rows and columns of the given hamiltonian
        row, col = H.shape[0], H.shape[1]

        # The hamiltonian must be a squared-one with 2**n x 2**n entries
        n_row, n_col = np.log2(row), np.log2(col)
        n = int(np.ceil(max(n_row, n_col)))
        size = 1<<n
        if row != col or not n_row.is_integer() or not n_col.is_integer():
            # Matrix with 2**n x 2**n zeros
            if self.real_H:
                square_H = np.zeros((size, size))
            else:
                square_H = np.zeros((size, size), dtype=complex)
            # Overlap the original hamiltonian in the top-left corner
            square_H[:row, :col] = H
            H = square_H
        self.H = H

        # Compute rows
        self.rows = np.arange(1<<n)

        # If the matrix is diagonal, only sigma_0=I and sigma_3=sigma_z are
        # relevant
        flag_diag = False
        if (self.H == np.diag(np.diagonal(self.H))).all():
            iterable = [PAULI_LABELS[0], PAULI_LABELS[3]]
            flag_diag = True
        else:
            iterable = PAULI_LABELS

        # Compute possible combinations
        combs = it.product(iterable, repeat=n)
        # If all entries are real, avoid an odd number of sigma_2=sigma_y
        if self.sym:
            combs = filter(lambda x: x.count('Y') % 2 == 0, combs)

        # Store coefficients in a dictionary where the keys will be the labels
        # of the compositions and the values will be the associated constants
        with Manager() as manager:
            coefficients = manager.dict()
            rows = Array('i', self.rows, lock=False)
            if flag_diag:
                if self.real_H:
                    H = Array('f', np.diag(H), lock=False)
                    pool = Pool(
                        initializer=init_pool_diag_real,
                        initargs=(size, coefficients, rows, H))
                    pool.imap_unordered(compute_weights_diag_real, combs)
                else:
                    H_real = Array('f', np.diag(H.real), lock=False)
                    H_imag = Array('f', np.diag(H.imag), lock=False)
                    pool = Pool(
                        initializer=init_pool_diag,
                        initargs=(size, coefficients, rows, H_real, H_imag))
                    pool.imap_unordered(compute_weights_diag, combs)
            else:
                if self.real_H:
                    H = Array('f', (self.H).flatten(), lock=False)
                    pool = Pool(
                        initializer=init_pool_general_real,
                        initargs=(size, coefficients, rows, H))
                    pool.map(compute_weights_general_real, combs)
                else:
                    H_real = Array('f', H.real.flatten(), lock=False)
                    H_imag = Array('f', H.imag.flatten(), lock=False)
                    pool = Pool(
                        initializer=init_pool_general,
                        initargs=(size, coefficients, rows, H_real, H_imag))
                    pool.imap_unordered(compute_weights_general, combs)
            pool.close()
            pool.join()
            coefficients = dict(coefficients)

        self.coefficients = coefficients
