"""Train a goal-conditioned temporal metric on frozen flat LeWM latents."""

import os
import csv
import math
from itertools import islice
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Compose, build_hdf5_dataset, get_column_normalizer
from temporal_metric import (
    TemporalMetric,
    make_temporal_pairs,
    temporal_metric_loss,
)
from utils import resolve_model_checkpoint_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def strip_level1_suffix(batch):
    """Convert HDF5Dataset's ``*_level1`` batch keys to the flat JEPA keys."""
    out = {}
    for key, value in batch.items():
        if key.endswith("_level1"):
            out[key[: -len("_level1")]] = value
        else:
            out[key] = value
    return out


def normalize_uint8_pixels(batch):
    for key, value in list(batch.items()):
        if not key.startswith("pixels"):
            continue
        if not torch.is_tensor(value) or value.dtype != torch.uint8:
            continue
        view_shape = [1] * value.ndim
        view_shape[-3] = 3
        mean = value.new_tensor(IMAGENET_MEAN, dtype=torch.float32).view(*view_shape)
        std = value.new_tensor(IMAGENET_STD, dtype=torch.float32).view(*view_shape)
        batch[key] = value.float().div(255.0).sub(mean).div(std)
    return batch


def save_checkpoint(out_path, state_path, metric, optim, epoch):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metric, out_path)
    torch.save(
        {
            "epoch": epoch,
            "model": metric.state_dict(),
            "optim": optim.state_dict(),
        },
        state_path,
    )


def epoch_checkpoint_path(out_dir, output_name, epoch):
    """Per-epoch metric checkpoint path; epoch is 0-based."""
    return Path(out_dir) / (
        f"{Path(output_name).stem}_epoch_{epoch + 1:02d}.pt"
    )


@torch.no_grad()
def encode_metric_batch(world_model, batch, cfg):
    """Encode DROID frames in chunks using exactly the offline-eval image transform.

    The encoder is framewise; metric training never needs predictor/action outputs.
    Chunking only the encoder avoids feeding a 37-frame clip to the causal predictor.
    """
    if cfg.data.dataset.get("type", None) != "droid":
        return world_model.encode(normalize_uint8_pixels(batch))["embed_0"].detach()

    from droid_transforms import img_transform

    pixels = batch["pixels"]
    if pixels.dtype != torch.uint8:
        raise ValueError("DROID metric training expects raw uint8 pixels")
    b, t = pixels.shape[:2]
    flat_pixels = pixels.flatten(0, 1)
    chunk_size = int(cfg.metric.get("encode_batch_size", 32))
    if chunk_size < 1:
        raise ValueError("metric.encode_batch_size must be positive")
    tf = img_transform(int(cfg.img_size))
    embeddings = []
    for start in range(0, b * t, chunk_size):
        stop = start + chunk_size
        info = {"pixels": tf(flat_pixels[start:stop])[:, None]}
        if "proprio" in batch:
            info["proprio"] = batch["proprio"].flatten(0, 1)[start:stop, None]
        embeddings.append(world_model.encode(info)["embed_0"][:, 0])
    return torch.cat(embeddings).reshape(b, t, -1).detach()


def training_batch_limit(loader_length, limit=None):
    """Match Lightning's positive integer/fraction limit_train_batches semantics."""
    if limit is None:
        return loader_length
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        return min(loader_length, limit)
    if isinstance(limit, float) and 0 < limit <= 1:
        return min(loader_length, max(1, math.floor(loader_length * limit)))
    raise ValueError("trainer.limit_train_batches must be a positive int or float in (0, 1]")


@hydra.main(version_base=None, config_path="config/train", config_name=None)
def run(cfg: DictConfig):
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(cfg.seed)
    cache_dir = os.environ.get("HJEPA_HOME", None)

    dataset_cfg = cfg.data.dataset
    train_dataset = build_hdf5_dataset(dataset_cfg, cache_dir=cache_dir)
    extra_transforms = [
        get_column_normalizer(train_dataset, col, col)
        for col in cfg.data.dataset.keys_to_load
        if not col.startswith("pixels")
    ]
    train_dataset.transform = Compose(*extra_transforms)

    loader = DataLoader(train_dataset, **cfg.loader)
    if len(train_dataset) == 0:
        raise ValueError("metric training dataset has 0 samples")
    n_batches = training_batch_limit(len(loader), cfg.trainer.get("limit_train_batches"))
    if n_batches == 0:
        raise ValueError("metric training loader has 0 batches; reduce loader.batch_size or disable drop_last")

    import stable_worldmodel as swm

    world_model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir)
    world_model = world_model.to(device).eval().requires_grad_(False)

    dim = int(cfg.level1.wm.embed_dim)
    metric = TemporalMetric(
        dim=dim,
        hidden_dim=int(cfg.metric.get("hidden_dim", 256)),
        depth=int(cfg.metric.get("depth", 2)),
        dropout=float(cfg.metric.get("dropout", 0.0)),
    ).to(device)

    optim = torch.optim.AdamW(
        metric.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    ckpt_path = resolve_model_checkpoint_path(cfg.policy, cache_dir)
    out_dir = Path(ckpt_path).parent
    out_name = str(cfg.metric.get("output_name", "temporal_metric.pt"))
    out_path = out_dir / out_name
    state_path = out_dir / f"{Path(out_name).stem}_state.pt"

    print(
        f"device={device} dataset={len(train_dataset)} "
        f"batches/epoch={n_batches} out={out_path}",
        flush=True,
    )

    max_steps = int(cfg.metric.get("max_steps", 15))
    min_steps = int(cfg.metric.get("min_steps", 1))
    gap = int(cfg.metric.get("gap", 2))
    margin = float(cfg.metric.get("margin", 1.0))
    negative_weight = float(cfg.metric.get("negative_weight", 0.0))
    max_epochs = int(cfg.trainer.max_epochs)
    save_every_n_epochs = int(cfg.metric.get("save_every_n_epochs", 1))
    if max_epochs < 1 or save_every_n_epochs < 1:
        raise ValueError("max_epochs and metric.save_every_n_epochs must be positive")
    sampling = str(cfg.metric.get("sampling", "legacy"))
    triples_per_clip = int(cfg.metric.get("triples_per_clip", 1))
    if sampling not in {"legacy", "uniform_far"} or triples_per_clip < 1:
        raise ValueError("invalid metric.sampling or metric.triples_per_clip")
    log_path = out_dir / f"{Path(out_name).stem}_train.csv"
    stat_keys = ("loss", "d_near_mean", "d_far_mean", "rank_accuracy",
                 "margin_accuracy", "distance_gap_mean")
    gap_bins = (("short", 0, 4), ("medium", 5, 12), ("long", 13, max_steps))
    print(f"sampling={sampling} triples/clip={triples_per_clip} log={log_path}", flush=True)

    for epoch in range(max_epochs):
        totals = dict.fromkeys(stat_keys, 0.0)
        n_pairs = 0
        offset_totals = dict.fromkeys(("near_offset", "far_offset", "far_max", "anchor_start"), 0.0)
        grouped = {name: [0, 0] for name, _, _ in gap_bins}
        n_steps = 0
        pbar = tqdm(islice(loader, n_batches), total=n_batches, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            batch = strip_level1_suffix(batch)
            if "action" not in batch:
                raise KeyError(
                    "metric training batch has no 'action'; expected HDF5Dataset "
                    "keys such as 'action_level1' or 'action'"
                )
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            emb = encode_metric_batch(world_model, batch, cfg)

            pairs = make_temporal_pairs(
                emb,
                max_steps=max_steps,
                min_steps=min_steps,
                gap=gap,
                sampling=sampling,
                triples_per_clip=triples_per_clip,
            )
            d_near = metric(pairs["anchor"], pairs["near"])
            d_far = metric(pairs["anchor"], pairs["far"])
            d_neg = None
            if negative_weight > 0.0:
                if emb.size(0) < 2:
                    raise ValueError("negative_weight > 0 requires at least two clips per batch")
                # Roll clips, not flattened triples: avoid same-clip false negatives.
                negatives = pairs["near"].reshape(emb.size(0), triples_per_clip, -1)
                negatives = negatives.roll(1, 0).reshape_as(pairs["near"])
                d_neg = metric(pairs["anchor"], negatives)

            stats = temporal_metric_loss(
                d_near,
                d_far,
                margin=margin,
                d_negative=d_neg,
                negative_weight=negative_weight,
            )
            optim.zero_grad()
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 1.0)
            optim.step()

            count = d_near.numel()
            for key in stat_keys:
                totals[key] += stats[key].detach().item() * count
            n_pairs += count
            offset_totals["near_offset"] += pairs["k_near"].sum().item()
            offset_totals["far_offset"] += pairs["k_far"].sum().item()
            max_k = min(max_steps, emb.size(1) - 1)
            offset_totals["far_max"] += (pairs["k_far"] == max_k).sum().item()
            offset_totals["anchor_start"] += (pairs["t"] == 0).sum().item()
            temporal_gap = pairs["k_far"] - pairs["k_near"]
            correct = d_far.detach() > d_near.detach()
            for name, low, high in gap_bins:
                mask = (temporal_gap >= low) & (temporal_gap <= high)
                grouped[name][0] += (correct & mask).sum().item()
                grouped[name][1] += mask.sum().item()
            n_steps += 1
            if step % int(cfg.metric.get("log_every", 10)) == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{totals['loss'] / n_pairs:.4f}",
                        "near": f"{stats['d_near_mean'].item():.3f}",
                        "far": f"{stats['d_far_mean'].item():.3f}",
                        "rank_acc": f"{totals['rank_accuracy'] / n_pairs:.3f}",
                        "margin_acc": f"{totals['margin_accuracy'] / n_pairs:.3f}",
                        "d_gap": f"{totals['distance_gap_mean'] / n_pairs:.3f}",
                    }
                )
        row = {"epoch": epoch + 1, "steps": n_steps, "triples": n_pairs}
        row.update({key: value / n_pairs for key, value in totals.items()})
        row.update({key: value / n_pairs for key, value in offset_totals.items()})
        for name, (correct_count, count) in grouped.items():
            row[f"{name}_rank_accuracy"] = correct_count / count if count else ""
            row[f"{name}_count"] = count
        with log_path.open("w" if epoch == 0 else "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"epoch {epoch + 1}: " + " ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"), flush=True)
        # Keep numbered weights and latest model/optimizer state on the same cadence.
        # Always preserve the final epoch, even if the interval does not divide it.
        if (epoch + 1) % save_every_n_epochs == 0 or epoch + 1 == max_epochs:
            save_checkpoint(out_path, state_path, metric, optim, epoch)
            epoch_path = epoch_checkpoint_path(out_dir, out_name, epoch)
            torch.save(metric, epoch_path)
            print(f"saved epoch {epoch + 1} metric to {epoch_path}", flush=True)

    print(f"saved temporal metric to {out_path}", flush=True)


if __name__ == "__main__":
    run()
