import numpy as np

def token_embedding_forward(token_ids, embed_table):
    # token_ids: (B, S) int
    # embed_table: (V, D) float
    # returns: (B, S, D) cache
    embeddings = embed_table[token_ids]
    cache = (token_ids, embed_table.shape)
    return embeddings, cache



def token_embedding_backward(grad_out, cache):
    # grad_out: (B, S, D)
    # returns: (grad_table, ), (v, D)
    token_ids, table_shape = cache
    grad_table = np.zeros(table_shape, dtype=grad_out.dtype)
    np.add.at(grad_table, token_ids, grad_out)
    return (grad_table,)

def positional_embedding_forward(seq_len, pos_table):
    # seq_len: int
    # pos_table: (max_seq_len, D)
    # returns: (seq_len, D), cache
    pos_emb = pos_table[:seq_len]
    cache = (seq_len, pos_table.shape)
    return pos_emb, cache

def positional_embedding_backward(grad_out, cache):
    # grad_out: (seq_len, D)
    # returns: (grad_pos_table , ) of shape (max_seq_len, D)
    seq_len, table_shape = cache
    grad_table = np.zeros(table_shape, dtype=grad_out.dtype)
    grad_table[:seq_len] = grad_out
    return (grad_table,)

