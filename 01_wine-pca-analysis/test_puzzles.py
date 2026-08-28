import numpy as np
import pytest

from puzzles import (
    rotate_90,
    transpose_no_T,
    is_magic_square,
    block_trace,
    top_k_indices,
)


# ROTATE 90

def test_rotate_90_basic():
    m = np.array([[1, 2],
                  [3, 4]])
    np.testing.assert_array_equal(
        rotate_90(m),
        np.array([[3, 1],
                  [4, 2]])
    )


def test_rotate_90_single():
    m = np.array([[42]])
    np.testing.assert_array_equal(rotate_90(m), np.array([[42]]))


def test_rotate_90_shape():
    m = np.arange(9).reshape(3, 3)
    res = rotate_90(m)
    assert res.shape == (3, 3)


def test_rotate_90_invariant_4_times():
    m = np.random.randint(0, 10, (5, 5))
    res = rotate_90(rotate_90(rotate_90(rotate_90(m))))
    np.testing.assert_array_equal(res, m)



# TRANSPOSE (no .T)


def test_transpose_basic():
    m = np.array([[1, 2, 3],
                  [4, 5, 6]])
    np.testing.assert_array_equal(
        transpose_no_T(m),
        np.array([[1, 4],
                  [2, 5],
                  [3, 6]])
    )


def test_transpose_single():
    m = np.array([[7]])
    np.testing.assert_array_equal(transpose_no_T(m), np.array([[7]]))


def test_transpose_rectangular_full():
    m = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10]])

    expected = np.array([
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [5, 10]
    ])

    np.testing.assert_array_equal(transpose_no_T(m), expected)


def test_transpose_invariant():
    m = np.random.randint(0, 10, (4, 6))
    np.testing.assert_array_equal(
        transpose_no_T(transpose_no_T(m)),
        m
    )


# MAGIC SQUARE

def test_magic_square_true():
    m = np.array([[2, 7, 6],
                  [9, 5, 1],
                  [4, 3, 8]])
    assert is_magic_square(m)


def test_magic_square_false():
    m = np.array([[1, 2],
                  [3, 4]])
    assert not is_magic_square(m)


def test_magic_square_non_square_2d():
    m = np.array([[1, 2, 3]])
    assert not is_magic_square(m)


def test_magic_square_non_2d():
    m = np.array([1, 2, 3])
    assert not is_magic_square(m)


def test_magic_square_modified():
    m = np.array([[2, 7, 6],
                  [9, 5, 1],
                  [4, 3, 9]])  # broken
    assert not is_magic_square(m)


#  BLOCK TRACE

def test_block_trace_basic():
    m = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(
        block_trace(m, 2),
        np.array([[5, 9],
                  [21, 25]])
    )


def test_block_trace_identity():
    m = np.eye(4)
    np.testing.assert_array_equal(
        block_trace(m, 2),
        np.array([[2, 0],
                  [0, 2]])
    )


def test_block_trace_k_equals_n():
    m = np.arange(9).reshape(3, 3)
    np.testing.assert_array_equal(
        block_trace(m, 3),
        np.array([[0]])
    )


def test_block_trace_invalid_k():
    m = np.arange(9).reshape(3, 3)
    with pytest.raises(Exception):
        block_trace(m, 2)


def test_block_trace_random_invariant():
    n, k = 6, 2
    m = np.random.randint(0, 10, (n, n))
    res = block_trace(m, k)

    # shape check
    assert res.shape == (n // k, n // k)


# TOP-K INDICES

def test_top_k_basic():
    v = np.array([1, 5, 3, 9, 2])
    np.testing.assert_array_equal(
        top_k_indices(v, 2),
        np.array([3, 1])
    )


def test_top_k_all():
    v = np.array([4, 1, 7])
    np.testing.assert_array_equal(
        top_k_indices(v, 3),
        np.array([2, 0, 1])
    )


def test_top_k_single():
    v = np.array([10, 20, 5])
    np.testing.assert_array_equal(
        top_k_indices(v, 1),
        np.array([1])
    )


def test_top_k_ties():
    v = np.array([5, 1, 5, 2])
    res = top_k_indices(v, 2)

    # must contain both max elements
    assert set(res) == {0, 2}


def test_top_k_ordering():
    v = np.array([1, 100, 50, 99])
    res = top_k_indices(v, 3)

    # ensure indices correspond to descending values
    assert v[res[0]] >= v[res[1]] >= v[res[2]]