"""Resume training from the most recent run's ckpt_latest.npz.

Finds the most recently modified subdirectory in runs/, reads the
run_args.json saved there, and re-launches scripts/train.py with
--resume_from pointing at ckpt_latest.npz.

Usage
-----
    python scripts/resume_training.py [extra train.py args...]

Any extra arguments are appended to the reconstructed command, allowing
overrides (e.g. --total_steps 200000).
"""
import json
import os
import subprocess
import sys


RUNS_DIR = "runs"


def find_latest_run():
    """Return the path to ckpt_latest.npz in the most recently modified run dir."""
    if not os.path.isdir(RUNS_DIR):
        print(f"No '{RUNS_DIR}/' directory found. Start a fresh training run first.")
        sys.exit(1)

    candidates = []
    for name in os.listdir(RUNS_DIR):
        run_path = os.path.join(RUNS_DIR, name)
        ckpt     = os.path.join(run_path, "ckpt_latest.npz")
        if os.path.isdir(run_path) and os.path.isfile(ckpt):
            candidates.append((os.path.getmtime(run_path), run_path, ckpt))

    if not candidates:
        print(f"No run directories with ckpt_latest.npz found in '{RUNS_DIR}/'.")
        sys.exit(1)

    candidates.sort(reverse=True)
    _, run_dir, ckpt_path = candidates[0]
    return run_dir, ckpt_path


def main():
    run_dir, ckpt_path = find_latest_run()
    print(f"Found run: {run_dir}")
    print(f"Resuming from: {ckpt_path}")

    args_path = os.path.join(run_dir, "run_args.json")
    if not os.path.isfile(args_path):
        print(f"No run_args.json in {run_dir} — cannot reconstruct training args.")
        print(f"Run manually with: python scripts/train.py --resume_from {ckpt_path}")
        sys.exit(1)

    with open(args_path) as f:
        saved = json.load(f)

    # Build command, forwarding original args and overriding resume_from
    cmd = [sys.executable, "scripts/train.py"]
    skip_keys = {"resume_from", "run_dir"}
    for key, val in saved.items():
        if key in skip_keys:
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif val is not None:
            cmd += [flag, str(val)]

    cmd += ["--resume_from", ckpt_path,
            "--run_dir",     run_dir]

    # Append any extra CLI args passed to this script
    cmd += sys.argv[1:]

    print(f"\nCommand:\n  {' '.join(cmd)}\n")
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
