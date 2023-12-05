"""
Pauli(Diag)Composer class definition.

See: https://arxiv.org/abs/2301.00560
"""

import warnings
import numpy as np
import scipy.sparse as ss
from numbers import Number

from utils import BINARY


# Ignore ComplexWarning
warnings.simplefilter('ignore', np.ComplexWarning)

class PauliComposer:

    def __init__(self, entry: str, weight: Number = None):
        # Compute the number of dimensions for the given entry
        n = len(entry)
        self.n = n

        # Compute some helpful powers
        self.dim = 1<<n

        # Store the entry converting the Pauli labels into uppercase
        self.entry = entry.upper()
        self.paulis = list(set(self.entry))

        # (-i)**(0+4m)=1, (-i)**(1+4m)=-i, (-i)**(2+4m)=-1, (-i)**(3+4m)=i
        mat_ent = {0: 1, 1: -1j, 2: -1, 3: 1j}

        # Count the number of ny mod 4
        self.ny = self.entry.count('Y') & 3
        init_ent = mat_ent[self.ny]
        if weight is not None:
            # first non-zero entry
            init_ent *= weight
        self.init_entry = init_ent
        self.iscomplex = np.iscomplex(init_ent)

        # Reverse the input and its 'binary' representation
        rev_entry = self.entry[::-1]
        rev_bin_entry = ''.join([BINARY[ent] for ent in rev_entry])

        # Column of the first-row non-zero entry
        col_val = int(''.join([BINARY[ent] for ent in self.entry]), 2)

        # Initialize an empty (2**n x 3)-matrix (rows, columns, entries)
        # row = np.arange(self.dim) 
        col = np.empty(self.dim, dtype=np.int32)
        # FIXME: storing rows and columns as np.complex64 since NumPy arrays
        # must have the same data type for each entry. Consider using
        # pd.DataFrame?

        col[0] = col_val  # first column
        # The AND bit-operator computes more rapidly mods of 2**n. Check that:
        #    x mod 2**n == x & (2**n-1)
        if weight is not None:
            if self.iscomplex:
                ent = np.full(self.dim, self.init_entry)
            else:
                ent = np.full(self.dim, float(self.init_entry))
        else:
            if self.iscomplex:
                ent = np.full(self.dim, self.init_entry, dtype=np.complex64)
            else:
                ent = np.full(self.dim, self.init_entry, dtype=np.int8)

        for ind in range(n):
            p = 1<<int(ind)  # left-shift of bits ('1' (1) << 2 = '100' (4))
            p2 = p<<1
            disp = p if rev_bin_entry[ind] == '0' else -p  # displacements
            col[p:p2] = col[0:p] + disp  # compute new columns
            # col[p:p2] = col[0:p] ^ p  # alternative for computing column

            # Store the new entries using old ones
            if rev_entry[ind] in ['I', 'X']:
                ent[p:p2] = ent[0:p]
            else:
                ent[p:p2] = -ent[0:p]

        self.col = col
        self.mat = ent

    def to_sparse(self):
        self.row = np.arange(self.dim)
        return ss.csr_matrix((self.mat, (self.row, self.col)),
                             shape=(self.dim, self.dim))

    def to_matrix(self):
        return self.to_sparse().toarray()


class PauliDiagComposer:

    def __init__(self, entry: str, weight: Number = None):
        # Compute the number of dimensions for the given entry
        n = len(entry)
        self.n = n

        # Compute some helpful powers
        self.dim = 1<<n

        # Store the entry converting the Pauli labels into uppercase
        self.entry = entry.upper()

        # Reverse the input and its 'binary' representation
        rev_entry = self.entry[::-1]

        # FIXME: storing rows and columns as np.complex64 since NumPy arrays
        # must have the same data type for each entry. Consider using
        # pd.DataFrame?

        # mat[:, 0] = mat[:, 1] = np.arange(self.dim)  # rows, columns
        if weight is not None:
            # first non-zero entry
            mat = np.full(self.dim, weight)
        else:
            mat = np.ones(self.dim, dtype=np.int8)

        for ind in range(n):
            p = 1<<int(ind)  # left-shift of bits ('1' (1) << 2 = '100' (4))
            p2 = p<<1
            # Store the new entries using old ones
            if rev_entry[ind] == 'I':
                mat[p:p2] = mat[0:p]
            else:
                mat[p:p2] = -mat[0:p]

        self.mat = mat

    def to_sparse(self):
        return ss.csr_matrix((self.mat, (np.arange(self.dim), np.arange(self.dim))),
                             shape=(self.dim, self.dim))

    def to_matrix(self):
        return self.to_sparse().toarray()
