"""Full encoder-decoder transformer model: wires encoder, decoder, and output projection."""

import numpy as np

from picochem.embeddings import (
    token_embedding_forward, positional_embedding_forward,
)
from picochem.encoder import encoder_block_forward
from picochem.decoder import decoder_block_forward
from picochem.ops import layer_norm_forward


def init_params(config, rng):
    """Initialize all model parameters."""
    D = config['d_model']
    DF = config['d_ff']
    init_scale = 0.02

    def randn(*shape):
        return rng.standard_normal(shape).astype(np.float64) * init_scale

    def zeros(*shape):
        return np.zeros(shape, dtype=np.float64)

    def ones(*shape):
        return np.ones(shape, dtype=np.float64)

    def init_encoder_block():
        b = {
            'ln1_gamma': ones(D), 'ln1_beta': zeros(D),
            'ln2_gamma': ones(D), 'ln2_beta': zeros(D),
            'attn_Wq': randn(D, D), 'attn_Wk': randn(D, D),
            'attn_Wv': randn(D, D), 'attn_Wo': randn(D, D),
            'attn_bq': zeros(D), 'attn_bk': zeros(D),
            'attn_bv': zeros(D), 'attn_bo': zeros(D),
            'ffn_W1': randn(D, DF), 'ffn_b1': zeros(DF),
            'ffn_W2': randn(DF, D), 'ffn_b2': zeros(D),
        }
        return b

    def init_decoder_block():
        b = {
            'ln1_gamma': ones(D), 'ln1_beta': zeros(D),
            'ln2_gamma': ones(D), 'ln2_beta': zeros(D),
            'ln3_gamma': ones(D), 'ln3_beta': zeros(D),
            'ffn_W1': randn(D, DF), 'ffn_b1': zeros(DF),
            'ffn_W2': randn(DF, D), 'ffn_b2': zeros(D),
        }
        for prefix in ['self', 'cross']:
            for w in ['Wq', 'Wk', 'Wv', 'Wo']:
                b[f'{prefix}_{w}'] = randn(D, D)
            for bs in ['bq', 'bk', 'bv', 'bo']:
                b[f'{prefix}_{bs}'] = zeros(D)
        return b

    return {
        'src_token_embed': randn(config['src_vocab'], D),
        'tgt_token_embed': randn(config['tgt_vocab'], D),
        'src_pos_embed': randn(config['max_src_len'], D),
        'tgt_pos_embed': randn(config['max_tgt_len'], D),
        'encoder_blocks': [init_encoder_block() for _ in range(config['n_enc_layers'])],
        'decoder_blocks': [init_decoder_block() for _ in range(config['n_dec_layers'])],
        'final_ln_gamma': ones(D),
        'final_ln_beta': zeros(D),
    }


def make_padding_mask(token_mask):
    """(B, S) 0/1 mask -> (B, 1, 1, S) additive mask."""
    return (1.0 - token_mask)[:, None, None, :] * -1e9


def make_causal_mask(seq_len):
    """(1, 1, T, T) upper-triangular -inf mask."""
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float64), k=1)
    return mask[None, None, :, :]


def model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, config):
    """Full encoder-decoder forward pass."""
    B, S = src_ids.shape
    _, T = tgt_ids.shape

    # Encoder embeddings
    src_tok_emb, src_tok_cache = token_embedding_forward(src_ids, params['src_token_embed'])
    src_pos_emb, src_pos_cache = positional_embedding_forward(S, params['src_pos_embed'])
    enc_input = src_tok_emb + src_pos_emb[None, :, :]

    enc_padding_mask = make_padding_mask(src_mask)

    # Encoder stack
    enc_x = enc_input
    enc_block_caches = []
    for block_params in params['encoder_blocks']:
        enc_x, block_cache = encoder_block_forward(
            enc_x, block_params, config['n_heads'], padding_mask=enc_padding_mask,
        )
        enc_block_caches.append(block_cache)
    encoder_output = enc_x

    # Decoder embeddings
    tgt_tok_emb, tgt_tok_cache = token_embedding_forward(tgt_ids, params['tgt_token_embed'])
    tgt_pos_emb, tgt_pos_cache = positional_embedding_forward(T, params['tgt_pos_embed'])
    dec_input = tgt_tok_emb + tgt_pos_emb[None, :, :]

    causal = make_causal_mask(T)
    dec_padding_mask = make_padding_mask(tgt_mask)
    decoder_self_mask = causal + dec_padding_mask

    # Decoder stack
    dec_x = dec_input
    dec_block_caches = []
    for block_params in params['decoder_blocks']:
        dec_x, block_cache = decoder_block_forward(
            dec_x, encoder_output, block_params, config['n_heads'],
            causal_mask=decoder_self_mask,
            encoder_padding_mask=enc_padding_mask,
        )
        dec_block_caches.append(block_cache)

    # Final layer norm
    dec_normed, final_ln_cache = layer_norm_forward(
        dec_x, params['final_ln_gamma'], params['final_ln_beta'],
    )

    # Output projection (weight tied with target token embedding)
    dec_flat = dec_normed.reshape(B * T, config['d_model'])
    logits_flat = dec_flat @ params['tgt_token_embed'].T
    logits = logits_flat.reshape(B, T, config['tgt_vocab'])

    cache = {
        'src_ids': src_ids, 'tgt_ids': tgt_ids,
        'src_tok_cache': src_tok_cache, 'src_pos_cache': src_pos_cache,
        'tgt_tok_cache': tgt_tok_cache, 'tgt_pos_cache': tgt_pos_cache,
        'enc_block_caches': enc_block_caches,
        'dec_block_caches': dec_block_caches,
        'final_ln_cache': final_ln_cache,
        'dec_normed': dec_normed, 'dec_flat': dec_flat,
        'B': B, 'S': S, 'T': T,
    }
    return logits, cache

from picochem.embeddings import (
    token_embedding_backward, positional_embedding_backward,
)
from picochem.encoder import encoder_block_backward
from picochem.decoder import decoder_block_backward
from picochem.ops import layer_norm_backward, softmax_cross_entropy_forward, softmax_cross_entropy_backward


def compute_loss(logits, targets, ignore_index=-1):
    B, T, V = logits.shape 
    logits_flat = logits.reshape(B*T, V) 
    targets_flat = targets.reshape(B*T) 
    loss, ce_cache = softmax_cross_entropy_forward(logits_flat, targets_flat, ignore_index) 
    cache = (ce_cache, B, T, V) 
    return loss, cache 

def loss_backward(grad_loss, cache): 
    ce_cache, B, T, V = cache 
    grad_logits_flat, _ = softmax_cross_entropy_backward(grad_loss, ce_cache) 
    grad_logits = grad_logits_flat.reshape(B, T, V) 
    return grad_logits 

def model_backward(grad_logits, cache, params, config):
    B = cache['B']
    T = cache['T']
    S = cache['S']
    D = config['d_model']
    grads = {}

    # Output projection: logits_flat = dec_flat @ tgt_token_embed.T
    grad_logits_flat = grad_logits.reshape(B * T, config['tgt_vocab'])
    dec_flat = cache['dec_flat']
    grad_dec_flat = grad_logits_flat @ params['tgt_token_embed']        # (B*T, D)
    grad_tgt_token_embed_from_proj = grad_logits_flat.T @ dec_flat     # (V, D)
    grad_dec_normed = grad_dec_flat.reshape(B, T, D)

    # Final layer norm
    grad_dec_x, grad_final_ln_gamma, grad_final_ln_beta = layer_norm_backward(
        grad_dec_normed, cache['final_ln_cache']
    )
    grads['final_ln_gamma'] = grad_final_ln_gamma
    grads['final_ln_beta'] = grad_final_ln_beta

    # Decoder blocks (reversed)
    grads['decoder_blocks'] = [None] * config['n_dec_layers']
    grad_encoder_output = np.zeros((B, S, D), dtype=np.float64)
    for i in reversed(range(config['n_dec_layers'])):
        grad_dec_x, grad_enc_contribution, block_grads = decoder_block_backward(
            grad_dec_x, cache['dec_block_caches'][i]
        )
        grad_encoder_output += grad_enc_contribution
        grads['decoder_blocks'][i] = block_grads

    # Target embeddings: dec_input = tgt_tok_emb + tgt_pos_emb[None]
    grad_tgt_token_embed_from_input, = token_embedding_backward(
        grad_dec_x, cache['tgt_tok_cache']
    )
    grad_tgt_pos_embed, = positional_embedding_backward(
        grad_dec_x.sum(axis=0), cache['tgt_pos_cache']
    )
    grads['tgt_token_embed'] = grad_tgt_token_embed_from_input + grad_tgt_token_embed_from_proj
    grads['tgt_pos_embed'] = grad_tgt_pos_embed

    # Encoder blocks (reversed)
    grads['encoder_blocks'] = [None] * config['n_enc_layers']
    grad_enc_x = grad_encoder_output
    for i in reversed(range(config['n_enc_layers'])):
        grad_enc_x, block_grads = encoder_block_backward(
            grad_enc_x, cache['enc_block_caches'][i]
        )
        grads['encoder_blocks'][i] = block_grads

    # Source embeddings: enc_input = src_tok_emb + src_pos_emb[None]
    grad_src_token_embed, = token_embedding_backward(
        grad_enc_x, cache['src_tok_cache']
    )
    grad_src_pos_embed, = positional_embedding_backward(
        grad_enc_x.sum(axis=0), cache['src_pos_cache']
    )
    grads['src_token_embed'] = grad_src_token_embed
    grads['src_pos_embed'] = grad_src_pos_embed

    return grads

# Greedy Decoder
def greedy_decode(src_ids, src_mask, params, config,
                  start_token, end_token, pad_token,
                  max_length=200):
    # Algorithm: 
    output = [start_token]
    for _ in range(max_length - 1): 
        # Current decoder input (1, T) 
        tgt_ids = np.array([output], dtype=np.int32)
        tgt_mask = np.ones_like(tgt_ids, dtype=np.float64) 
        # Forward pass - only the last pos logits 
        logits, _ = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, config)
        next_logits = logits[0, -1] 

        # Greedy: pick argmax
        next_token = int(np.argmax(next_logits)) 
        output.append(next_token) 

        if next_token == end_token: 
            break 
    return output

def sample_decode(src_ids, src_mask, params, config,
                  start_token, end_token, pad_token,
                  max_length=200, temperature=1.0, rng=None):
    """Generate using temperature sampling.

    temperature < 1.0: more conservative (close to greedy)
    temperature = 1.0: sample from raw distribution
    temperature > 1.0: more random
    """
    if rng is None:
        rng = np.random.default_rng()

    output = [start_token]

    for _ in range(max_length - 1):
        tgt_ids = np.array([output], dtype=np.int32)
        tgt_mask = np.ones_like(tgt_ids, dtype=np.float64)

        logits, _ = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, config)
        next_logits = logits[0, -1]

        # Apply temperature and sample
        scaled = next_logits / temperature
        probs = np.exp(scaled - scaled.max())  # stable softmax
        probs = probs / probs.sum()
        next_token = int(rng.choice(len(probs), p=probs))

        output.append(next_token)

        if next_token == end_token:
            break

    return output