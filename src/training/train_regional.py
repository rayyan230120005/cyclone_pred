"""
Regional Cyclone Model Trainer.

Trains the two learned components of the system for one ocean basin:
  - MultimodalCycloneClassifier   intensity category + wind + pressure + RMW
  - CycloneTrajectoryConvLSTM     48-hour track and intensity projection

Supervision comes from IBTrACS best-track observations via
`src.data_pipeline.cyclone_dataset.RealCycloneDataset` - real storms, real winds,
real observed motion. Nothing here is synthesized.

What the training loop does:
  - reads configs/hyperparams.yaml (CLI flags override individual fields)
  - holds out the most recent seasons as validation, split by season so no storm
    can appear on both sides
  - class-weighted focal loss for the badly imbalanced IMD categories
  - Haversine (great-circle) loss on the track, in kilometres, so the objective
    matches the metric the system is actually judged on
  - Gaussian NLL for the uncertainty cone, learned from the model's own track
    residuals rather than regressed against an invented target
  - masks every loss term that depends on an unobserved value
  - mixed precision, gradient clipping, linear warmup into cosine decay
  - keeps the checkpoint with the best validation track error, and early-stops

Usage:
    python -m src.training.train_regional --basin bay_of_bengal --epochs 60
    python -m src.training.train_regional --basin bay_of_bengal --imagery-root data/raw/hursat
    python -m src.training.train_regional --all-basins --device cuda
"""

import argparse
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.data_pipeline.cyclone_dataset import IMD_CLASSES, RealCycloneDataset
from src.models.classification_fusion import MultimodalCycloneClassifier
from src.models.prediction_convlstm import CycloneTrajectoryConvLSTM

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# Regression targets live on very different numeric scales (wind ~20-140 kts,
# pressure ~900-1010 hPa). Losses are computed on values divided by these scales so
# no single head dominates the gradient. Model outputs stay in physical units, so
# inference and ONNX export are unaffected.
WIND_SCALE_KTS = 30.0
PRESSURE_SCALE_HPA = 20.0
RMW_SCALE_KM = 25.0


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss with optional per-class weights.

    IMD categories are severely imbalanced - super cyclonic storms are well under 1%
    of observations. Plain cross-entropy is minimised by a model that never predicts
    the severe classes, which are the only ones that matter operationally.
    """

    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if self.class_weights is not None else None
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def haversine_km(
    lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor
) -> torch.Tensor:
    """
    Great-circle distance in kilometres between two sets of coordinates, in degrees.

    Differentiable, so it can be optimised directly. This matters: one degree of
    longitude is ~111 km at the equator but ~104 km at 20 N, so an L1 loss on raw
    lat/lon deltas silently mis-weights east-west error against north-south error.
    """
    lat1r, lon1r = torch.deg2rad(lat1), torch.deg2rad(lon1)
    lat2r, lon2r = torch.deg2rad(lat2), torch.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(torch.clamp(a, min=1e-12)))


class TrajectoryLoss(nn.Module):
    """
    Track loss over the forecast horizon.

    The decoder emits per-step [dlat, dlon, wind, raw_sigma]. Deltas are accumulated
    into absolute positions and scored by great-circle distance against the observed
    track. The fourth channel is read as log-sigma of the position error and trained
    by Gaussian negative log-likelihood, so the uncertainty cone is calibrated on the
    model's own residuals - wide where the model is genuinely unsure, tight where it
    is not - instead of being regressed onto a made-up radius.
    """

    def __init__(self, wind_weight: float = 0.3, nll_weight: float = 0.2):
        super().__init__()
        self.wind_weight = wind_weight
        self.nll_weight = nll_weight

    def forward(
        self,
        pred: torch.Tensor,        # [B, T, 4] predicted per-step deltas
        target: torch.Tensor,      # [B, T, 4] observed per-step deltas
        mask: torch.Tensor,        # [B, T]    1 where the step was observed
        start_lat: torch.Tensor,   # [B]
        start_lon: torch.Tensor,   # [B]
    ) -> Dict[str, torch.Tensor]:
        denom = mask.sum().clamp(min=1.0)

        # Accumulate incremental deltas into absolute track positions.
        pred_lat = start_lat[:, None] + torch.cumsum(pred[..., 0], dim=1)
        pred_lon = start_lon[:, None] + torch.cumsum(pred[..., 1], dim=1)
        true_lat = start_lat[:, None] + torch.cumsum(target[..., 0] * mask, dim=1)
        true_lon = start_lon[:, None] + torch.cumsum(target[..., 1] * mask, dim=1)

        dist_km = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
        track_loss = (dist_km * mask).sum() / denom

        # Intensity along the track.
        wind_err = (pred[..., 2] - target[..., 2]).abs() / WIND_SCALE_KTS
        wind_loss = (wind_err * mask).sum() / denom

        # Calibrated spread: Gaussian NLL of the observed track error under the
        # model's predicted sigma. log_sigma is clamped for numerical safety.
        log_sigma = torch.clamp(pred[..., 3], min=math.log(5.0), max=math.log(800.0))
        sigma = torch.exp(log_sigma)
        nll = log_sigma + 0.5 * (dist_km.detach() / sigma) ** 2
        nll_loss = (nll * mask).sum() / denom

        total = (
            track_loss / 100.0                      # normalise km into a ~unit range
            + self.wind_weight * wind_loss
            + self.nll_weight * nll_loss
        )
        return {
            "total": total,
            "track_km": (dist_km * mask).sum().detach() / denom,
            "wind": wind_loss.detach(),
            "nll": nll_loss.detach(),
        }


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale-normalised smooth L1, applied only where the target was actually observed.
    Returns a zero that still carries grad when nothing in the batch is observed.
    """
    per_sample = F.smooth_l1_loss(pred / scale, target / scale, reduction="none").squeeze(-1)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------
class RegionalTrainer:
    """
    Trains and validates one basin's model pair, keeping the best checkpoint.
    """

    def __init__(
        self,
        basin: str = "bay_of_bengal",
        config: Optional[Dict[str, Any]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
        imagery_root: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_train_samples: Optional[int] = None,
    ):
        self.basin = basin
        self.cfg = config or {}
        train_cfg = self.cfg.get("training", {})
        model_cfg = self.cfg.get("models", {})

        self.epochs = epochs if epochs is not None else int(train_cfg.get("epochs", 60))
        self.batch_size = batch_size if batch_size is not None else int(train_cfg.get("batch_size", 32))
        self.lr = lr if lr is not None else float(train_cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))
        self.grad_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
        self.patience = int(train_cfg.get("early_stopping_patience", 10))
        self.num_workers = num_workers if num_workers is not None else int(train_cfg.get("num_workers", 4))
        want_amp = bool(train_cfg.get("mixed_precision", True))

        self.device = self._resolve_device(device)
        self.use_amp = want_amp and self.device.type == "cuda"
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ---------------- data ----------------
        imagery = "hursat" if imagery_root else "none"
        common = dict(
            basin=basin,
            imagery=imagery,
            imagery_root=imagery_root,
            val_seasons=int(self.cfg.get("data", {}).get("val_seasons", 6)),
        )
        self.train_ds = RealCycloneDataset(split="train", augment=True, **common)
        self.val_ds = RealCycloneDataset(split="val", augment=False, **common)

        if max_train_samples and max_train_samples < len(self.train_ds):
            # Smoke-test path: a fixed, reproducible subset.
            g = torch.Generator().manual_seed(int(self.cfg.get("project", {}).get("seed", 42)))
            idx = torch.randperm(len(self.train_ds), generator=g)[:max_train_samples].tolist()
            self.train_ds = torch.utils.data.Subset(self.train_ds, idx)

        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=pin, drop_last=len(self.train_ds) > self.batch_size,
            persistent_workers=self.num_workers > 0,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=pin,
            persistent_workers=self.num_workers > 0,
        )

        # ---------------- models ----------------
        sat_cfg = model_cfg.get("classification_fusion", {}).get("satellite_branch", {})
        pretrained = bool(sat_cfg.get("pretrained", True)) and imagery != "none"

        self.classifier = MultimodalCycloneClassifier(
            in_satellite_channels=4, synoptic_features_dim=8, pretrained=pretrained
        ).to(self.device)
        self.convlstm = CycloneTrajectoryConvLSTM(in_channels=4, forecast_steps=8).to(self.device)

        # ---------------- losses ----------------
        base_ds = self.train_ds.dataset if isinstance(self.train_ds, torch.utils.data.Subset) else self.train_ds
        weights = base_ds.class_weights().to(self.device)
        self.focal_loss = FocalLoss(class_weights=weights).to(self.device)

        lw = self.cfg.get("loss_weights", {})
        self.w_cls = float(lw.get("classification_focal", 1.0))
        self.w_wind = float(lw.get("msw_mae", 0.5))
        self.w_press = float(lw.get("pressure_mae", 0.3))
        self.w_traj = float(lw.get("trajectory_haversine", 1.5))
        self.trajectory_loss = TrajectoryLoss(nll_weight=float(lw.get("uncertainty_nll", 0.2)))

        # ---------------- optimiser ----------------
        self.params = list(self.classifier.parameters()) + list(self.convlstm.parameters())
        self.optimizer = optim.AdamW(self.params, lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        logger.info(
            f"[{basin}] device={self.device} amp={self.use_amp} pretrained={pretrained} "
            f"train={len(self.train_ds)} val={len(self.val_ds)} imagery={imagery}"
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    def _lr_at(self, epoch: int) -> float:
        """Linear warmup into cosine decay."""
        if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            return self.lr * epoch / max(self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.min_lr + (self.lr - self.min_lr) * cosine

    def _forward_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared forward pass for training and validation."""
        dev = self.device
        img = batch["satellite_image"].to(dev, non_blocking=True)
        syn = batch["synoptic_vector"].to(dev, non_blocking=True)
        seq = batch["sequence_tensor"].to(dev, non_blocking=True)
        labels = batch["class_label"].to(dev, non_blocking=True)
        t_wind = batch["wind_speed"].to(dev, non_blocking=True)
        t_press = batch["central_pressure"].to(dev, non_blocking=True)
        t_rmw = batch["rmw"].to(dev, non_blocking=True)
        t_traj = batch["future_trajectory"].to(dev, non_blocking=True)
        traj_mask = batch["trajectory_mask"].to(dev, non_blocking=True)
        press_obs = batch["pressure_observed"].to(dev, non_blocking=True)
        rmw_obs = batch["rmw_observed"].to(dev, non_blocking=True)
        lat0 = batch["current_lat"].to(dev, non_blocking=True)
        lon0 = batch["current_lon"].to(dev, non_blocking=True)

        out_cls = self.classifier(img, syn)
        loss_cls = self.focal_loss(out_cls["intensity_logits"], labels)
        ones = torch.ones_like(press_obs)
        loss_wind = masked_l1(out_cls["wind_speed_kts"], t_wind, ones, WIND_SCALE_KTS)
        loss_press = masked_l1(out_cls["central_pressure_hpa"], t_press, press_obs, PRESSURE_SCALE_HPA)
        loss_rmw = masked_l1(out_cls["rmw_km"], t_rmw, rmw_obs, RMW_SCALE_KM)

        # Seed the decoder with the storm's current state: [dlat, dlon, wind, log_sigma].
        seed = torch.cat(
            [
                torch.zeros_like(t_wind),
                torch.zeros_like(t_wind),
                t_wind,
                torch.full_like(t_wind, math.log(40.0)),
            ],
            dim=1,
        )
        out_traj = self.convlstm(seq, seed)
        traj = self.trajectory_loss(out_traj, t_traj, traj_mask, lat0, lon0)

        total = (
            self.w_cls * loss_cls
            + self.w_wind * loss_wind
            + self.w_press * loss_press
            + 0.2 * loss_rmw
            + self.w_traj * traj["total"]
        )

        with torch.no_grad():
            preds = out_cls["intensity_logits"].argmax(dim=1)
            correct = (preds == labels).float().sum()
            wind_mae = (out_cls["wind_speed_kts"] - t_wind).abs().mean()

        return {
            "loss": total,
            "cls": loss_cls.detach(),
            "wind": loss_wind.detach(),
            "press": loss_press.detach(),
            "traj_km": traj["track_km"],
            "nll": traj["nll"],
            "correct": correct,
            "wind_mae": wind_mae.detach(),
            "n": torch.tensor(float(labels.size(0)), device=dev),
        }

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.classifier.train(train)
        self.convlstm.train(train)

        agg: Dict[str, float] = {}
        n_batches = 0
        n_samples = 0.0

        for batch in loader:
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self._forward_losses(batch)
                self.scaler.scale(out["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                # Clip across BOTH models - the ConvLSTM is the unstable half.
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self._forward_losses(batch)

            n_batches += 1
            n_samples += float(out["n"].item())
            for key in ("loss", "cls", "wind", "press", "traj_km", "nll", "wind_mae"):
                agg[key] = agg.get(key, 0.0) + float(out[key].item())
            agg["correct"] = agg.get("correct", 0.0) + float(out["correct"].item())

        if n_batches == 0:
            return {}

        metrics = {k: round(v / n_batches, 4) for k, v in agg.items() if k != "correct"}
        metrics["accuracy"] = round(agg["correct"] / max(n_samples, 1.0), 4)
        return metrics

    def train(self) -> Dict[str, Any]:
        logger.info(f"[{self.basin}] training for {self.epochs} epochs")
        history: List[Dict[str, Any]] = []
        best_track_km = float("inf")
        best_epoch = 0
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.basin}_best.pth")

        for epoch in range(1, self.epochs + 1):
            lr_now = self._lr_at(epoch)
            for group in self.optimizer.param_groups:
                group["lr"] = lr_now

            t0 = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "lr": round(lr_now, 8),
                "elapsed_sec": round(elapsed, 2),
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            logger.info(
                f"[{self.basin}] epoch {epoch}/{self.epochs} "
                f"({elapsed:.1f}s, lr={lr_now:.2e}) | "
                f"train loss {train_metrics.get('loss', 0):.4f} "
                f"acc {train_metrics.get('accuracy', 0):.3f} | "
                f"val loss {val_metrics.get('loss', 0):.4f} "
                f"acc {val_metrics.get('accuracy', 0):.3f} "
                f"track {val_metrics.get('traj_km', 0):.1f} km "
                f"wind MAE {val_metrics.get('wind_mae', 0):.1f} kts"
            )

            # Selection metric is mean validation track error - the number the
            # system is ultimately judged on.
            current = val_metrics.get("traj_km", float("inf"))
            if current < best_track_km:
                best_track_km = current
                best_epoch = epoch
                torch.save(
                    {
                        "classifier_state": self.classifier.state_dict(),
                        "convlstm_state": self.convlstm.state_dict(),
                        "basin": self.basin,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "train_metrics": train_metrics,
                        "imd_classes": IMD_CLASSES,
                        "config": self.cfg,
                    },
                    ckpt_path,
                )
                logger.info(f"[{self.basin}] new best ({current:.1f} km) -> {ckpt_path}")

            if epoch - best_epoch >= self.patience:
                logger.info(
                    f"[{self.basin}] early stop: no improvement in {self.patience} epochs "
                    f"(best {best_track_km:.1f} km at epoch {best_epoch})"
                )
                break

        history_path = os.path.join(self.checkpoint_dir, f"{self.basin}_history.json")
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        logger.info(
            f"[{self.basin}] done. best val track error {best_track_km:.1f} km "
            f"at epoch {best_epoch}. checkpoint: {ckpt_path}"
        )
        return {
            "basin": self.basin,
            "checkpoint_path": ckpt_path,
            "history_path": history_path,
            "best_epoch": best_epoch,
            "best_val_track_km": round(best_track_km, 2),
            "history": history,
        }


# ----------------------------------------------------------------------------
def load_config(path: str = os.path.join("configs", "hyperparams.yaml")) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.warning(f"Config not found at {path}; using built-in defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regional cyclone models on IBTrACS.")
    parser.add_argument("--basin", default="bay_of_bengal",
                        choices=["bay_of_bengal", "arabian_sea", "indian_ocean_south"])
    parser.add_argument("--all-basins", action="store_true", help="Train every basin in sequence")
    parser.add_argument("--config", default=os.path.join("configs", "hyperparams.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--imagery-root", default=None, help="HURSAT-B1 NetCDF directory")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training samples - use for a fast smoke test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)

    seed = int(cfg.get("project", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    basins = ["bay_of_bengal", "arabian_sea", "indian_ocean_south"] if args.all_basins else [args.basin]

    results = []
    for basin in basins:
        trainer = RegionalTrainer(
            basin=basin,
            config=cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            imagery_root=args.imagery_root,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
        )
        results.append(trainer.train())

    print("\n=== Training summary ===")
    for res in results:
        print(
            f"  {res['basin']:<20} best val track error "
            f"{res['best_val_track_km']:>7.1f} km  (epoch {res['best_epoch']})  "
            f"-> {res['checkpoint_path']}"
        )


if __name__ == "__main__":
    main()
