"""GPU (device-resident) training loop for picochem.

Runs the full transformer forward+backward on the GPU via picochem.device_layers
(every op on resident DeviceTensors). The transformer stack params live on the
GPU and are updated in place with dt_adam; the embedding tables stay on the host
(gather + scatter) and use the NumPy Adam — the same host/device split the
device-layer tests use.

Requires the CUDA extension built:  bash scripts/build_cuda.sh
Run with picochem/kernels on the path:
    PYTHONPATH=picochem/kernels python scripts/train_device.py --synthetic --total_steps 100
    PYTHONPATH=picochem/kernels python scripts/train_device.py --data data/traces.parquet ...
"""
import argparse
import datetime
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "picochem/kernels")

import picochem_cuda as pc
import picochem.device_layers as dl
from picochem.checkpointing import save_checkpoint
from picochem.model import init_params, make_padding_mask, make_causal_mask
from picochem.optimizer import init_adam_state, adam_step

DT = pc.DeviceTensor


def _materialize(mask_b1xx, B, H, T, S):
    """(B,1,*,S)-style additive mask -> (B*H, T, S) float32 DeviceTensor."""
    full = np.broadcast_to(mask_b1xx, (B, H, T, S)).reshape(B * H, T, S).astype(np.float32)
    return DT(full)


def main():
    ap = argparse.ArgumentParser(description="Device-resident GPU training for picochem.")
    ap.add_argument("--data", default=None, help="traces.parquet; omit with --synthetic")
    ap.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    ap.add_argument("--iupac_vocab", default="data/iupac_vocab.json")
    ap.add_argument("--synthetic", action="store_true",
                    help="train to overfit one fixed random batch (smoke test, no data needed)")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--n_enc_layers", type=int, default=3)
    ap.add_argument("--n_dec_layers", type=int, default=3)
    ap.add_argument("--max_src_len", type=int, default=64)
    ap.add_argument("--max_tgt_len", type=int, default=64)
    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=1000,
                    help="save a numpy checkpoint every N steps (0 disables)")
    ap.add_argument("--run_dir", default=None, help="checkpoint dir (default runs/device_<ts>)")
    ap.add_argument("--src_vocab", type=int, default=64, help="synthetic only")
    ap.add_argument("--tgt_vocab", type=int, default=80, help="synthetic only")
    args = ap.parse_args()

    if args.run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join("runs", f"device_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)

    H = args.n_heads
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(0)

    # ── data ────────────────────────────────────────────────────────────────
    if args.synthetic:
        src_vocab, tgt_vocab = args.src_vocab, args.tgt_vocab
        B, S, T = args.batch_size, 24, 20
        fixed = dict(
            src=rng.integers(3, src_vocab, (B, S)).astype(np.int32),
            tin=rng.integers(3, tgt_vocab, (B, T)).astype(np.int32),
            tout=rng.integers(3, tgt_vocab, (B, T)).astype(np.int32),
            sm=np.ones((B, S)), tm=np.ones((B, T)),
        )
        def get_batch():
            return fixed['src'], fixed['tin'], fixed['tout'], fixed['sm'], fixed['tm']
    else:
        from picochem.data import load_vocab
        from picochem.data_loader import load_dataset, split_dataset, make_batch
        sv, _ = load_vocab(args.smiles_vocab)
        iv, _ = load_vocab(args.iupac_vocab)
        src_vocab, tgt_vocab = len(sv), len(iv)
        src_pad, tgt_pad = sv["<pad>"], iv["<pad>"]
        pairs = load_dataset(args.data, sv, iv, args.max_src_len, args.max_tgt_len)
        train_pairs, _ = split_dataset(pairs)
        print(f"train pairs: {len(train_pairs):,}")
        def get_batch():
            return make_batch(train_pairs, args.batch_size, src_pad, tgt_pad, rng)

    cfg = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, d_model=args.d_model,
               n_heads=H, d_ff=args.d_ff, n_enc_layers=args.n_enc_layers,
               n_dec_layers=args.n_dec_layers, max_src_len=args.max_src_len,
               max_tgt_len=args.max_tgt_len)
    p = init_params(cfg, np.random.default_rng(0))

    # ── resident stack params + device Adam state ─────────────────────────────
    enc = [{k: dl.to_dt(v) for k, v in bp.items()} for bp in p['encoder_blocks']]
    dec = [{k: dl.to_dt(v) for k, v in bp.items()} for bp in p['decoder_blocks']]
    fg, fb = dl.to_dt(p['final_ln_gamma']), dl.to_dt(p['final_ln_beta'])
    stack = {'enc': enc, 'dec': dec, 'fg': fg, 'fb': fb}
    mstate = {'enc': [{k: dl.zeros_like_dt(v) for k, v in b.items()} for b in enc],
              'dec': [{k: dl.zeros_like_dt(v) for k, v in b.items()} for b in dec],
              'fg': dl.zeros_like_dt(fg), 'fb': dl.zeros_like_dt(fb)}
    vstate = {'enc': [{k: dl.zeros_like_dt(v) for k, v in b.items()} for b in enc],
              'dec': [{k: dl.zeros_like_dt(v) for k, v in b.items()} for b in dec],
              'fg': dl.zeros_like_dt(fg), 'fb': dl.zeros_like_dt(fb)}

    # embedding tables stay on host; their Adam state is host too
    emb = {k: p[k].copy() for k in ('src_token_embed', 'tgt_token_embed', 'src_pos_embed', 'tgt_pos_embed')}
    emb_state = init_adam_state(emb)

    def adam_stack(g, step):
        for name, blocks in (('encoder_blocks', enc), ('decoder_blocks', dec)):
            mb = mstate['enc'] if name == 'encoder_blocks' else mstate['dec']
            vb = vstate['enc'] if name == 'encoder_blocks' else vstate['dec']
            for i, blk in enumerate(blocks):
                for k in blk:
                    pc.dt_adam(blk[k], g[name][i][k], mb[i][k], vb[i][k], step, args.lr, b1, b2, eps)
        pc.dt_adam(fg, g['final_ln_gamma'], mstate['fg'], vstate['fg'], step, args.lr, b1, b2, eps)
        pc.dt_adam(fb, g['final_ln_beta'], mstate['fb'], vstate['fb'], step, args.lr, b1, b2, eps)

    def snapshot_and_save(step):
        """Download resident params + host embeddings into a numpy params dict
        (matching init_params' structure) and save a checkpoint."""
        np_p = {
            'src_token_embed': emb['src_token_embed'], 'tgt_token_embed': emb['tgt_token_embed'],
            'src_pos_embed': emb['src_pos_embed'], 'tgt_pos_embed': emb['tgt_pos_embed'],
            'encoder_blocks': [{k: blk[k].numpy() for k in blk} for blk in enc],
            'decoder_blocks': [{k: blk[k].numpy() for k in blk} for blk in dec],
            'final_ln_gamma': fg.numpy(), 'final_ln_beta': fb.numpy(),
        }
        path = os.path.join(args.run_dir, "ckpt_latest.npz")
        save_checkpoint(path, np_p, {}, step, cfg)

    print(f"Device training: {args.total_steps} steps, batch {args.batch_size}, "
          f"d_model {args.d_model}, {args.n_enc_layers}+{args.n_dec_layers} layers, "
          f"vocab {src_vocab}/{tgt_vocab}\n")
    t0 = time.perf_counter()
    for step in range(1, args.total_steps + 1):
        src, tin, tout, sm, tm = get_batch()
        B, S = src.shape
        _, T = tin.shape

        src_emb = (emb['src_token_embed'][src] + emb['src_pos_embed'][:S]).astype(np.float32)
        tgt_emb = (emb['tgt_token_embed'][tin] + emb['tgt_pos_embed'][:T]).astype(np.float32)
        enc_mask = _materialize(make_padding_mask(sm), B, H, S, S)
        self_mask = _materialize(make_causal_mask(T) + make_padding_mask(tm), B, H, T, T)
        cross_mask = _materialize(make_padding_mask(sm), B, H, T, S)
        tgt_embed_dt = dl.to_dt(emb['tgt_token_embed'])

        logits, cache = dl.model_forward(DT(src_emb), DT(tgt_emb), enc, dec, fg, fb,
                                         tgt_embed_dt, H, enc_mask, self_mask, cross_mask)
        loss, n_valid = pc.dt_cross_entropy_forward(logits, tout.reshape(-1).astype(np.int32), -1)
        grad_logits = pc.dt_cross_entropy_backward(logits, tout.reshape(-1).astype(np.int32), -1, n_valid, 1.0)
        g = dl.model_backward(grad_logits, cache)

        adam_stack(g, step)

        # embedding tables: scatter grads on host, then NumPy Adam
        gse = g['grad_src_emb'].numpy(); gte = g['grad_tgt_emb'].numpy()
        g_emb = {k: np.zeros_like(emb[k]) for k in emb}
        np.add.at(g_emb['src_token_embed'], src, gse)
        np.add.at(g_emb['tgt_token_embed'], tin, gte)
        g_emb['tgt_token_embed'] += g['grad_tgt_embed_proj'].numpy()
        g_emb['src_pos_embed'][:S] = gse.sum(0)
        g_emb['tgt_pos_embed'][:T] = gte.sum(0)
        adam_step(emb, g_emb, emb_state, step, lr=args.lr, beta1=b1, beta2=b2, eps=eps)

        if step % args.log_every == 0 or step == 1:
            dt_ms = (time.perf_counter() - t0) / step * 1000
            print(f"step {step:5d}  loss {float(loss):.4f}  ({dt_ms:.1f} ms/step)", flush=True)

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            snapshot_and_save(step)
            print(f"  checkpoint -> {args.run_dir}/ckpt_latest.npz", flush=True)

    snapshot_and_save(args.total_steps)
    print(f"\nDone. {args.total_steps} steps in {time.perf_counter()-t0:.1f}s  "
          f"(checkpoint in {args.run_dir})")


if __name__ == "__main__":
    main()
