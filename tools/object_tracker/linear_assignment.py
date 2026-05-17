"""
Create: 2026.05.04
Author: SG.SUH
Python: 3.8.13
PyTorch: 1.12.0
"""

import numpy as np

def linear_assignment(X):
    indices = _hungarian(X).tolist()
    indices.sort()

    indices = np.array(indices, dtype=int)

    indices.shape = (-1, 2)

    return indices

class _HungarianState(object):
    def __init__(self, cost_matrix):
        cost_matrix = np.atleast_2d(cost_matrix)

        transposed = (cost_matrix.shape[1] < cost_matrix.shape[0])

        if transposed:
            self.C = (cost_matrix.T).copy()
        else:
            self.C = cost_matrix.copy()

        self.transposed = transposed

        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=np.bool)
        self.col_uncovered = np.ones(m, dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _find_prime_in_row(self, row):
        col = np.argmax(self.marked[row] == 2)

        if self.marked[row, col] != 2:
            col = -1

        return col
    
    def _clear_covers(self):
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True

def _hungarian(cost_matrix):
    state = _HungarianState(cost_matrix)

    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    results = np.array(np.where(state.marked == 1)).T

    if state.transposed:
        results = results[:, ::-1]

    return results

def _step1(state):
    state.C -= state.C.min(axis=1)[:, np.newaxis]

    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()

    return _step3

def _step3(state):
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4
    
def _step4(state):
    C = (state.C == 0).astype(np.int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= state.col_uncovered.astype(dtype=np.int, copy=False)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))

        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2

            star_col = np.argmax(state.marked[row] == 1)

            if not state.marked[row, star_col] == 1:
                state.Z0_r = row
                state.Z0_c = col

                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (state.row_uncovered.astype(dtype=np.int, copy=False))
                covered_C[row] = 0

def _step5(state):
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        row = np.argmax(state.marked[:, path[count, 1]] == 1)

        if not state.marked[row, path[count, 1]] == 1:
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        col = np.argmax(state.marked[path[count, 0]] == 2)

        if state.marked[row, col] != 2:
            col = -1

        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()

    state.marked[state.marked == 2] = 0

    return _step3

def _step6(state):
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[np.logical_not(state.row_uncovered)] += minval
        state.C[:, state.col_uncovered] -= minval

    return _step4