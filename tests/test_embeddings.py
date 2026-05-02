"""Tests for embedding ops."""

import sys
sys.path.insert(0, ".")

import numpy as np

from picochem.embeddings import (
    token_embedding_forward, token_embedding_backward,
    positional_embedding_forward, positional_embedding_backward,
)


def test_token_embedding_forward_correctness():
    table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    token_ids = np.array([[0, 2], [1, 0]])

    out, _ = token_embedding_forward(token_ids, table)

    expected = np.array([[[1.0, 2.0], [5.0, 6.0]],
                         [[3.0, 4.0], [1.0, 2.0]]])
    np.testing.assert_array_equal(out, expected)


def test_token_embedding_backward_accumulates_repeated():
    """Same token appearing twice should accumulate gradients."""
    table_shape = (3, 2)
    token_ids = np.array([[0, 0]])  # token 0 appears twice
    grad_out = np.array([[[1.0, 2.0], [3.0, 4.0]]])

    cache = (token_ids, table_shape)
    grad_table, = token_embedding_backward(grad_out, cache)

    # Row 0 should have summed gradients from both positions
    expected = np.array([[4.0, 6.0], [0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(grad_table, expected)


def test_positional_embedding_forward():
    table = np.arange(20.0).reshape(10, 2)
    out, _ = positional_embedding_forward(3, table)
    np.testing.assert_array_equal(out, table[:3])


def test_positional_embedding_backward():
    table_shape = (10, 2)
    grad_out = np.ones((3, 2))
    cache = (3, table_shape)

    grad_table, = positional_embedding_backward(grad_out, cache)

    assert grad_table.shape == table_shape
    np.testing.assert_array_equal(grad_table[:3], np.ones((3, 2)))
    np.testing.assert_array_equal(grad_table[3:], np.zeros((7, 2)))