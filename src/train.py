"""Train scribe on TinyStories — epoch-based, pausable and resumable.

    python -m src.train --prepare --until-epoch 3   # download, then train to epoch 3
    python -m src.train --epochs 2                  # train 2 MORE epochs from where it left off
    python -m src.train --until-epoch 5             # keep going until 5 total
    python -m src.train --fresh --until-epoch 3     # ignore any checkpoint, start over
    # Ctrl-C at any time: finishes the current step, saves checkpoints/last.pt, exits.
    #   just run again to resume.

Resume is automatic whenever checkpoints/last.pt exists. Checkpoints hold the
full state (model, optimizer, epoch, global step, RNG), and are written every
epoch, every eval_interval steps, and on Ctrl-C — so a pause loses at most a few
hundred steps, and the LR schedule (keyed off global step) resumes exactly.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import signal
import time

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from src.data import make_loader, prepare
from src.model import GPT

LAST = "last.pt"      # rolling resumable checkpoint
BEST = "best.pt"      # lowest val loss seen


# Ctrl-C -> graceful pause. First press requests a save-and-exit at the next
# step boundary; a second press exits immediately.
_STOP = {"flag": False}


def _install_pause_handler() -> None:
    def handler(signum, frame):
        if _STOP["flag"]:
            raise KeyboardInterrupt
        _STOP["flag"] = True
        print("\n[pause requested — will save and exit after this step; Ctrl-C again to force]",
              flush=True)
    signal.signal(signal.SIGINT, handler)


def lr_at(step: int, total_steps: int, tc: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr over `total_steps`."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step >= total_steps:
        return tc.min_lr
    ratio = (step - tc.warmup_steps) / max(1, total_steps - tc.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return tc.min_lr + coeff * (tc.lr - tc.min_lr)


@torch.no_grad()
def estimate_loss(model, tc: TrainConfig, val_loader) -> float:
    if val_loader is None:
        return float("nan")
    model.eval()
    losses, it = [], iter(val_loader)
    for _ in range(tc.eval_iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(tc.device), y.to(tc.device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def save_ckpt(path, model, optim, gc, epoch, global_step, best_val):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "config": gc.__dict__,
        "epoch": epoch,               # epochs fully completed
        "global_step": global_step,
        "best_val": best_val,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="download + tokenize before training")
    ap.add_argument("--limit", type=int, default=None, help="cap #stories when preparing")
    ap.add_argument("--until-epoch", type=int, default=None, help="train until this total epoch")
    ap.add_argument("--epochs", type=int, default=None, help="train this many MORE epochs")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="train until this global step; also sets the cosine LR horizon "
                         "(so the schedule fully decays by then). Overrides epoch targets.")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    ap.add_argument("--smoke", action="store_true", help="tiny model + tiny data to test the loop")
    args = ap.parse_args()

    gc, tc = GPTConfig(), TrainConfig()
    if args.smoke:
        gc.n_layer, gc.n_head, gc.n_embd, gc.block_size = 2, 2, 128, 64
        tc.batch_size, tc.grad_accum, tc.epochs = 8, 1, 1
        tc.eval_interval, tc.eval_iters, tc.warmup_steps = 50, 20, 10
        if args.limit is None:
            args.limit = 2000

    _install_pause_handler()
    os.makedirs(tc.out_dir, exist_ok=True)
    print(f"device={tc.device} dtype={tc.dtype}")

    # ---- data ----
    train_bin = os.path.join(tc.data_dir, "train.bin")
    if args.smoke:
        if args.prepare or not os.path.exists(train_bin):
            import shutil
            prepare(tc.dataset_repo, tc.data_dir, "val", limit=args.limit)
            shutil.copyfile(os.path.join(tc.data_dir, "val.bin"), train_bin)
    elif args.prepare or not os.path.exists(train_bin):
        prepare(tc.dataset_repo, tc.data_dir, "val", limit=(args.limit or 5000))
        prepare(tc.dataset_repo, tc.data_dir, "train", limit=args.limit)

    train_loader = make_loader(tc.data_dir, "train", gc.block_size, tc.batch_size, shuffle=True)
    val_split = "val" if os.path.exists(os.path.join(tc.data_dir, "val.bin")) else None
    val_loader = make_loader(tc.data_dir, "val", gc.block_size, tc.batch_size, shuffle=False) if val_split else None
    steps_per_epoch = max(1, len(train_loader) // tc.grad_accum)

    # ---- model / optimizer ----
    model = GPT(gc).to(tc.device)
    optim = model.configure_optimizers(tc.weight_decay, tc.lr, (tc.beta1, tc.beta2))

    # ---- resume ----
    completed_epochs, global_step, best_val = 0, 0, float("inf")
    last_path = os.path.join(tc.out_dir, LAST)
    if os.path.exists(last_path) and not args.fresh:
        ck = torch.load(last_path, map_location=tc.device, weights_only=False)  # our own trusted ckpt
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        completed_epochs = ck["epoch"]
        global_step = ck["global_step"]
        best_val = ck.get("best_val", float("inf"))
        if "rng" in ck:
            torch.set_rng_state(ck["rng"]["torch"].cpu() if hasattr(ck["rng"]["torch"], "cpu") else ck["rng"]["torch"])
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["python"])
        print(f"resumed from {last_path}: {completed_epochs} epochs done, step {global_step}")
    else:
        torch.manual_seed(tc.seed); np.random.seed(tc.seed); random.seed(tc.seed)

    print(f"parameters: {model.num_params()/1e6:.2f}M | steps/epoch≈{steps_per_epoch}", flush=True)

    # ---- resolve target ----
    if args.max_steps is not None:
        # Step-based target: stop at max_steps AND aim the cosine schedule there,
        # so the LR fully decays to min_lr by then regardless of epoch length.
        total_steps = args.max_steps
        target_epoch = 10 ** 9  # effectively unbounded; the loop stops on global_step
        if global_step >= args.max_steps:
            print(f"already at step {global_step} (target {args.max_steps}); nothing to do.")
            return
        print(f"training to global step {args.max_steps} from {global_step} "
              f"(cosine horizon={total_steps})", flush=True)
    else:
        if args.until_epoch is not None:
            target_epoch = args.until_epoch
        elif args.epochs is not None:
            target_epoch = completed_epochs + args.epochs
        else:
            target_epoch = tc.epochs
        if target_epoch <= completed_epochs:
            print(f"already trained to epoch {completed_epochs} (target {target_epoch}); nothing to do.")
            return
        total_steps = target_epoch * steps_per_epoch  # cosine horizon aims at the target
        print(f"training epochs {completed_epochs+1}..{target_epoch}  (total_steps horizon={total_steps})", flush=True)

    # ---- train ----
    model.train()
    t0 = time.time()
    try:
        for epoch in range(completed_epochs, target_epoch):
            running = 0.0
            data_iter = iter(train_loader)
            for local_step in range(steps_per_epoch):
                for g in optim.param_groups:
                    g["lr"] = lr_at(global_step, total_steps, tc)

                optim.zero_grad(set_to_none=True)
                for _ in range(tc.grad_accum):
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        x, y = next(data_iter)
                    x, y = x.to(tc.device), y.to(tc.device)
                    _, loss = model(x, y)
                    (loss / tc.grad_accum).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optim.step()

                global_step += 1
                running += loss.item()
                if global_step % tc.log_interval == 0:
                    lr = optim.param_groups[0]["lr"]
                    denom = f"/{args.max_steps}" if args.max_steps else ""
                    print(f"  step {global_step}{denom} (epoch {epoch+1}) "
                          f"loss {loss.item():.4f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)

                # mid-epoch eval + checkpoint (also the pause-safety cadence)
                if global_step % tc.eval_interval == 0:
                    val = estimate_loss(model, tc, val_loader)
                    print(f"[eval] step {global_step} val_loss {val:.4f}", flush=True)
                    save_ckpt(last_path, model, optim, gc, epoch, global_step, best_val)
                    if val < best_val:
                        best_val = val
                        save_ckpt(os.path.join(tc.out_dir, BEST), model, optim, gc, epoch, global_step, best_val)

                if _STOP["flag"]:
                    save_ckpt(last_path, model, optim, gc, epoch, global_step, best_val)
                    print(f"[paused] saved {last_path} at epoch {epoch+1}, step {global_step}. "
                          f"Run again to resume.", flush=True)
                    return

                if args.max_steps is not None and global_step >= args.max_steps:
                    val = estimate_loss(model, tc, val_loader)
                    save_ckpt(last_path, model, optim, gc, epoch, global_step, best_val)
                    if val < best_val:
                        best_val = val
                        save_ckpt(os.path.join(tc.out_dir, BEST), model, optim, gc, epoch, global_step, best_val)
                    print(f"[done] reached max_steps {args.max_steps} at step {global_step} | "
                          f"val_loss {val:.4f} | best {best_val:.4f}", flush=True)
                    return

            # end of epoch
            completed_epochs = epoch + 1
            val = estimate_loss(model, tc, val_loader)
            print(f"=== epoch {completed_epochs}/{target_epoch} done | "
                  f"train_loss {running/steps_per_epoch:.4f} val_loss {val:.4f} ===", flush=True)
            save_ckpt(last_path, model, optim, gc, completed_epochs, global_step, best_val)
            if val < best_val:
                best_val = val
                save_ckpt(os.path.join(tc.out_dir, BEST), model, optim, gc, completed_epochs, global_step, best_val)
    except KeyboardInterrupt:
        save_ckpt(last_path, model, optim, gc, completed_epochs, global_step, best_val)
        print(f"\n[interrupted] saved {last_path}. Run again to resume.", flush=True)
        return

    print(f"done. reached epoch {completed_epochs}. best val loss {best_val:.4f}. "
          f"checkpoints in {tc.out_dir}/ (last.pt, best.pt)")


if __name__ == "__main__":
    main()
