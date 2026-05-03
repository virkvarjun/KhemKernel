"""Lightweight training logger — no external dependencies beyond matplotlib."""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless servers
import matplotlib.pyplot as plt


class TrainLogger:
    """Writes training metrics to log.jsonl, samples to samples.txt, and
    regenerates a loss_curve.png on every call to :meth:`plot`.

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

        # In-memory buffers for plotting (populated from disk on init so
        # resuming a run keeps the full history).
        self._steps  = []
        self._losses = []
        self._t0 = time.time()

        if os.path.exists(self._jsonl_path):
            with open(self._jsonl_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self._steps.append(entry["step"])
                    self._losses.append(entry["loss"])

    def log_step(self, step, loss, grad_norm, lr):
        """Append one line to log.jsonl and update the in-memory buffer."""
        entry = {
            "step":      step,
            "loss":      float(loss),
            "grad_norm": float(grad_norm),
            "lr":        float(lr),
            "time":      round(time.time() - self._t0, 2),
        }
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._steps.append(step)
        self._losses.append(float(loss))

    def log_sample(self, step, src, generated, target):
        """Append a generation sample to samples.txt."""
        with open(self._samples_path, "a") as f:
            f.write(f"--- step {step} ---\n")
            f.write(f"  src:       {src}\n")
            f.write(f"  generated: {generated}\n")
            f.write(f"  target:    {target}\n\n")

    def plot(self):
        """Regenerate loss_curve.png from all logged steps."""
        if len(self._steps) < 2:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self._steps, self._losses, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        if min(self._losses) > 0:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._plot_path, dpi=100)
        plt.close(fig)
