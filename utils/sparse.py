#! /usr/bin/env python
# author: kapil thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import numpy as np
from numpy import random
import operator


# A collection of staticmethods to support a view of a list as an
# two ordered arrays of the same length: one containing sorted indices of
# non-zero components and the other containing the corresponding
# component values.


def initialize(coarse=False):
    """Initialize an empty sparse vector.
    """
    dtype = np.float32 if coarse else np.float64
    return (np.array([], dtype=np.uint32), np.array([], dtype=dtype))


def to_list(sparse, size):
    """Produce a normal list of the given size from the sparse vector.
    """
    nonsparse = [0] * size
    for idx, value in zip(*sparse):
        nonsparse[idx] = value
    return list(nonsparse)


def from_list(nonsparse, **kwargs):
    """Produce a sparse vector representation from the given list where
    the sparse vector is a 2-tuple of lists of indices and their
    corresponding non-zero values.
    """
    components = [(idx, value) for idx, value in enumerate(nonsparse)
            if value != 0]
    return from_nonzero(components, **kwargs)


def from_nonzero(components, coarse=False):
    """Produce a sparse vector from a list of non-zero components where each
    component is represented as a 2-tuple (index, value).
    """
    if len(components) == 0:
        idxs, values = [], []
    else:
        idxs, values = zip(*components)

    dtype = np.float32 if coarse else np.float64
    return (np.array(idxs, dtype=np.uint32), np.array(values, dtype=dtype))


def to_nonzero(sparse):
    """Produce a list of non-zero components where each component is
    represented as a 2-tuple (index, value).
    """
    return zip(*sparse)


def is_sparse(vector):
    """Return whether the given vector appears to be a sparse vector or
    a regular list.
    """
    return len(vector) == 2 and \
            hasattr(vector[0], '__len__') and \
            hasattr(vector[1], '__len__') and \
            len(vector[0]) == len(vector[1])


def is_well_formed(vector):
    """Return whether the given sparse vector has non-repeating indices in
    ascending order.
    """
    last_idx = -1
    for idx in vector[0]:
        if last_idx >= idx:
            return False
        last_idx = idx
    return True


def num_nonzero(vector):
    """Return the number of non-zero components.
    """
    return len(vector[0])


def get_values(vector, idxs):
    """Return a list of values corresponding to the given list of sorted
    indices.
    """
    # Pre-sort just in case
    sorted_idx_pairs = sorted(enumerate(idxs), key=operator.itemgetter(1))
    sorted_idxs = [pair[1] for pair in sorted_idx_pairs]

    values = []
    v = 0
    v_limit = num_nonzero(vector)
    for i, idx in enumerate(sorted_idxs):
        while v < v_limit and idx > vector[0][v]:
            v += 1
        if v == v_limit:
            values.extend([0] * (len(idxs) - i))
            break
        if idx < vector[0][v]:
            values.append(0)
        elif idx == vector[0][v]:
            values.append(vector[1][v])

    unsorted_values = [values[pair[0]] for pair in sorted_idx_pairs]
    return unsorted_values


def update(vector, update_vector):
    """Update a list of components given a new sparse vector containing the
    new values. This update vector must include explicit 0 values for
    components that are being set to 0.
    """
    update_components = sorted(zip(*update_vector), key=operator.itemgetter(0))
    v = 0
    v_limit = num_nonzero(vector)
    final_components = []
    for i, component in enumerate(update_components):
        idx, val = component
        while v < v_limit and idx > vector[0][v]:
            final_components.append((vector[0][v], vector[1][v]))
            v += 1
        if v == v_limit:
            final_components.extend(component
                                 for component in update_components[i:]
                                 if component[1] != 0)
            break

        if val != 0:
            final_components.append((idx, val))
        if idx == vector[0][v]:
            v += 1

    final_components.extend(zip(vector[0][v:], vector[1][v:]))
    return from_nonzero(final_components)


def delete_idxs(vector, idxs):
    """Delete a list of components given their idxs in a sparse vector.
    """
    return update(vector, [(idx, 0) for idx in sorted(idxs)])


def offset_idxs(vector, offset):
    """Shift the indices of components by a given (positive) offset.
    """
    assert offset >= 0
    return (vector[0] + offset,
            vector[1])


def add_all(vectors):
    """Combine all sparse vectors into a single sparse vector, adding the
    components at the same index.
    """
    if len(vectors) == 1:
        return vectors[0]

    sums = defaultdict(int)
    for vector in vectors:
        for idx, value in zip(*vector):
            sums[idx] += value
    components = sorted(sums.iteritems(), key=operator.itemgetter(0))
    return from_nonzero(components)


def avg_all(vectors):
    """Return the average or centroid of a list of sparse vectors.
    """
    return scale(add_all(vectors), 1 / len(vectors))


def scale(vector, scalar):
    """Scale the entries in the sparse vector by the given scalar.
    """
    return (vector[0], vector[1] * scalar)


def dot(vector, weights):
    """Get a dot product of a sparse vector with a vector of weight
    parameters provided either as a sparse vector or a regular list.
    """
    #assert is_sparse(vector)
    if is_sparse(weights):
        w = 0
        w_limit = num_nonzero(weights)
        dot = 0
        for idx, value in zip(*vector):
            while w < w_limit and weights[0][w] < idx:
                w += 1
            if w == w_limit:
                break
            if weights[0][w] == idx:
                dot += value * weights[1][w]
        return dot
    else:
        try:
            return sum(value * weights[idx] for idx, value in zip(*vector))
        except TypeError:
            print len(vector)
            print type(vector)
            print vector


def add(vec0, vec1):
    """Add vec0 + vec1 where the inputs are sparse vectors or lists.
    """
    return pairwise(vec0, vec1, operator.__add__)


def subtract(vec0, vec1):
    """Subtract vec0 - vec1 where the inputs are sparse vectors or lists.
    """
    return pairwise(vec0, vec1, operator.__sub__)


def multiply(vec0, vec1):
    """Multiply vec0 .* vec1 where the inputs are sparse vectors or lists.
    """
    return pairwise(vec0, vec1, operator.__mul__)


def pairwise(vec0, vec1, oper):
    """Perform a pairwise operation on two sparse vectors or one sparse and
    one non-sparse vector. By convention, if either argument is a non-sparse
    list, it is returned as a list. If a sparse vector is needed, convert
    the input list to a sparse vector beforehand.
    """
    if is_sparse(vec0):
        if is_sparse(vec1):
            # Both are sparse vectors, so a sparse vector is returned
            return pairwise_sparse_sparse(vec0, vec1, oper)
        else:
            # vec1 is a list so a list is returned
            return pairwise_sparse_list(vec0, vec1, oper)
    else:
        if is_sparse(vec1):
            # vec0 is a list so a list is returned
            return pairwise_list_sparse(vec0, vec1, oper)
        else:
            # Both are lists. This is unexpected but we'll support this
            # to allow easy switching between sparse and non-sparse vectors.
            return oper(np.array(vec0), np.array(vec1))


def pairwise_sparse_sparse(vec0, vec1, oper):
    """Perform a pairwise operation on two sparse vectors and return a
    sparse vector.
    """
    v0, v1 = 0, 0
    v0_limit, v1_limit = num_nonzero(vec0), num_nonzero(vec1)
    final_components = []
    while v0 < v0_limit and v1 < v1_limit:
        # TODO: make lookups more efficient
        idx0, value0 = vec0[0][v0], vec0[1][v0]
        idx1, value1 = vec1[0][v1], vec1[1][v1]
        if idx0 == idx1:
            new_value = oper(value0, value1)
            if new_value != 0:
                final_components.append((idx0, new_value))
            v0 += 1
            v1 += 1
        elif idx0 < idx1:
            new_value = oper(value0, 0)
            if new_value != 0:
                final_components.append((idx0, new_value))
            v0 += 1
        else:
            new_value = oper(0, value1)
            if new_value != 0:
                final_components.append((idx1, new_value))
            v1 += 1

    while v0 < v0_limit:
        new_value = oper(vec0[1][v0], 0)
        if new_value != 0:
            final_components.append((vec0[0][v0], new_value))
        v0 += 1
    while v1 < v1_limit:
        new_value = oper(0, vec1[1][v1])
        if new_value != 0:
            final_components.append((vec1[0][v1], new_value))
        v1 += 1

    return from_nonzero(final_components)


def pairwise_sparse_list(vec0, list1, oper):
    """Perform a pairwise operation on a sparse vector and a list and return
    a list (numpy array).
    """
    v = 0
    v_limit = num_nonzero(vec0)
    new_list = oper(0, np.array(list1))

    idx0, value0 = None, None
    if v < v_limit:
        idx0, value0 = vec0[0][v], vec0[1][v]

    for idx1, value1 in enumerate(list1):
        if idx1 == idx0:
            new_list[idx1] = oper(value0, value1)
            v += 1
            if v < v_limit:
                idx0, value0 = vec0[0][v], vec0[1][v]
            else:
                break
    return new_list


def pairwise_list_sparse(list0, vec1, oper):
    """Perform a pairwise operation on a list and a sparse vector and return
    a list (numpy array).
    """
    v = 0
    v_limit = num_nonzero(vec1)
    new_list = oper(np.array(list0), 0)

    idx1, value1 = None, None
    if v < v_limit:
        idx1, value1 = vec1[0][v], vec1[1][v]

    for idx0, value0 in enumerate(list0):
        if idx0 == idx1:
            new_list[idx0] = oper(value0, value1)
            v += 1
            if v < v_limit:
                idx1, value1 = vec1[0][v], vec1[1][v]
            else:
                break
    return new_list


def random_list(length, sparsity=0.75, amplitude=1000, integers=True):
    """Get a random sparse list for tests.
    """
    rand_fn = random.randint if integers else random.uniform
    dense = rand_fn(-amplitude, amplitude, length)
    dense[random.random_sample(length) < sparsity] = 0
    return list(dense)


# TODO: convert these to real tests and check each function
if __name__ == '__main__':
    random_length = random.randint(10000)

    a_list = random_list(random_length)
    a_sparse = from_list(a_list)

    assert a_list == to_list(a_sparse, random_length)
    assert is_sparse(a_sparse)
    assert not is_sparse(a_list)

    # Pairwise operations
    b_list = random_list(random_length)
    b_sparse = from_list(b_list)

    al_plus_bl = add(a_list, b_list)
    al_plus_bs = add(a_list, b_sparse)
    as_plus_bl = add(a_sparse, b_list)
    as_plus_bs = add(a_sparse, b_sparse)

    assert all(al_plus_bl == al_plus_bs)
    assert all(al_plus_bl == as_plus_bl)
    assert all(al_plus_bl == to_list(as_plus_bs, random_length))
    assert zip(*from_list(al_plus_bl)) == zip(*as_plus_bs)

    al_minus_bl = subtract(a_list, b_list)
    al_minus_bs = subtract(a_list, b_sparse)
    as_minus_bl = subtract(a_sparse, b_list)
    as_minus_bs = subtract(a_sparse, b_sparse)

    assert all(al_minus_bl == al_minus_bs)
    assert all(al_minus_bl == as_minus_bl)
    assert all(al_minus_bl == to_list(as_minus_bs, random_length))
    assert zip(*from_list(al_minus_bl)) == zip(*as_minus_bs)

    # Dot products
    w_list = random_list(random_length, amplitude=10, sparsity=0.1,
                         integers=False)
    w_sparse = from_list(w_list)

    wl_dot_al = sum(w * a for w, a in zip(w_list, a_list))
    wl_dot_as = dot(a_sparse, w_list)
    ws_dot_as = dot(a_sparse, w_sparse)
    as_dot_ws = dot(w_sparse, a_sparse)

    assert wl_dot_al == wl_dot_as
    assert wl_dot_al == ws_dot_as
    assert wl_dot_al == as_dot_ws
