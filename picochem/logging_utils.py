"""Lightweight training logger — no external dependencies beyond matplotlib."""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt


class TrainLogger:
    """Writes training metrics to log.jsonl, samples to samples.txt, and
    regenerates plots on every call to :meth:`plot`.

    Produces two plot files on each :meth:`plot` call:
    - ``loss_curve.png`` — always written
    - ``training_progress.png`` — copy of the same figure

    Parameters
    ----------
    log_dir : str
        Directory to write all output files into (created if absent).
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._jsonl_path   = os.path.join(log_dir, "log.jsonl")
        self._samples_path = os.path.join(log_dir, "samples.txt")
        self._plot_path    = os.path.join(log_dir, "loss_curve.png")
        self._prog_path    = os.path.join(log_dir, "training_progress.png")

        # In-memory buffers — populated from disk so resuming keeps full history
        self._steps        = []
        self._losses       = []
        self._val_steps    = []
        self._val_losses   = []
        self._eval_steps   = []
        self._eval_matches = []
        self._t0 = time.time()

        if os.path.exists(self._jsonl_path):
            with open(self._jsonl_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = entry.get("type", "train")
                    if kind == "eval":
                        self._eval_steps.append(entry["step"])
                        self._eval_matches.append(entry.get("structure_match_rate", 0.0))
                    elif kind == "val":
                        self._val_steps.append(entry["step"])
                        self._val_losses.append(entry["loss"])
                    else:
                        self._steps.append(entry["step"])
                        self._losses.append(entry["loss"])

    # ── public methods ─────────────────────────────────────────────────────

    def log_step(self, step, loss, grad_norm, lr):
        """Append one training-loss line to log.jsonl."""
        entry = {
            "type":      "train",
            "step":      step,
            "loss":      float(loss),
            "grad_norm": float(grad_norm),
            "lr":        float(lr),
            "time":      round(time.time() - self._t0, 2),
        }
        self._write(entry)
        self._steps.append(step)
        self._losses.append(float(loss))

    def log_val(self, step, val_loss):
        """Append one validation-loss line to log.jsonl."""
        entry = {
            "type": "val",
            "step": step,
            "loss": float(val_loss),
            "time": round(time.time() - self._t0, 2),
        }
        self._write(entry)
        self._val_steps.append(step)
        self._val_losses.append(float(val_loss))

    def log_eval(self, step, validity, parse_rate, match_rate):
        """Append one OPSIN-eval line to log.jsonl."""
        entry = {
            "type":                  "eval",
            "step":                  step,
            "trace_validity_rate":   float(validity),
            "opsin_parse_rate":      float(parse_rate),
            "structure_match_rate":  float(match_rate),
            "time":                  round(time.time() - self._t0, 2),
        }
        self._write(entry)
        self._eval_steps.append(step)
        self._eval_matches.append(float(match_rate))

    def log_sample(self, step, src, generated, target):
        """Append a generation sample to samples.txt."""
        with open(self._samples_path, "a") as f:
            f.write(f"--- step {step} ---\n")
            f.write(f"  src:       {src}\n")
            f.write(f"  generated: {generated}\n")
            f.write(f"  target:    {target}\n\n")

    def plot(self):
        """Regenerate loss plots from all logged steps.

        Produces a 2-panel figure when both loss and eval data are available:
        - Top:    training loss (+ val loss overlay)
        - Bottom: structure match rate from OPSIN eval

        Falls back to a single-panel loss curve when eval data is absent.
        """
        if len(self._steps) < 2:
            return

        has_eval = len(self._eval_steps) >= 1

        if has_eval:
            fig, (ax_loss, ax_eval) = plt.subplots(
                2, 1, figsize=(9, 6), sharex=False
            )
        else:
            fig, ax_loss = plt.subplots(figsize=(9, 4))

        # ── train loss ────────────────────────────────────────────────────
        ax_loss.plot(self._steps, self._losses,
                     linewidth=0.8, label="train", alpha=0.8)
        if self._val_steps:
            ax_loss.plot(self._val_steps, self._val_losses,
                         linewidth=1.2, label="val", color="tab:orange")
        if min(self._losses) > 0:
            ax_loss.set_yscale("log")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, which="both", alpha=0.3)

        # ── structure match ───────────────────────────────────────────────
        if has_eval:
            ax_eval.plot(self._eval_steps, self._eval_matches,
                         marker="o", linewidth=1.2, color="tab:green")
            ax_eval.set_xlabel("Step")
            ax_eval.set_ylabel("Structure match rate")
            ax_eval.set_title("OPSIN Structure Match Rate")
            ax_eval.set_ylim(-0.05, 1.05)
            ax_eval.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._plot_path, dpi=100)
        fig.savefig(self._prog_path, dpi=100)
        plt.close(fig)

    # ── private ────────────────────────────────────────────────────────────

    def _write(self, entry):
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
