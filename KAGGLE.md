# Training on Kaggle GPUs

Free GPU training for the regional cyclone models. Kaggle gives ~30 GPU-hours/week, which is
far more than this project needs — a full three-basin run is well under an hour.

See [TRAINING.md](TRAINING.md) for what the training actually does. This file is only the
Kaggle-specific mechanics.

---

## Before you start: four Kaggle traps

These cost GPU quota if you hit them mid-run.

1. **Internet is OFF by default.** `download_ibtracs.py` and `git clone` both fail without it.
   Turn it on in the right-hand panel. Kaggle requires a phone-verified account to enable it.
2. **Do NOT run `pip install -r requirements.txt`.** Training needs only `numpy`, `pandas`,
   `requests`, `torch`, `torchvision`, `yaml` — every one already installed. The full
   requirements file pulls `tritonclient`, `streamlit`, `fastapi`, `cdsapi` and can replace
   Kaggle's CUDA torch with a CPU wheel, silently making training ~30× slower.
3. **Only `/kaggle/working` survives the session.** Anything written elsewhere is gone when
   the session ends, including your checkpoints.
4. **Pick GPU T4 ×2 or P100, not TPU.** The trainer is single-GPU unless you pass
   `--data-parallel` (see [Using both T4s](#using-both-t4s)); with T4 ×2 and no flag it uses
   the first card only. P100 is usually the faster single card.

---

## Step 1 — Create the notebook

1. https://www.kaggle.com/code → **New Notebook**
2. Right panel → **Accelerator** → `GPU T4 x2` (or `GPU P100`)
3. Right panel → **Internet** → **On**

---

## Step 2 — Verify the GPU

Always run this first. If it prints `False`, fix it before spending quota.

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("torch:", torch.__version__)
```

Expected: `CUDA: True`, `Tesla T4` or `Tesla P100-PCIE-16GB`.

The trainer calls `torch.amp.GradScaler("cuda")`, which needs **torch ≥ 2.4**. If Kaggle's
image is older, see [Troubleshooting](#troubleshooting).

---

## Step 3 — Get the code

**Public repo:**

```python
!git clone https://github.com/harsh-kulkarni-05/cyclone_pred.git /kaggle/working/cyclone_pred
%cd /kaggle/working/cyclone_pred
!git checkout feat/real-training-data
```

**Private repo** — do not paste a token into a cell. Add it via **Add-ons → Secrets** as
`GITHUB_TOKEN`, then:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/harsh-kulkarni-05/cyclone_pred.git /kaggle/working/cyclone_pred
%cd /kaggle/working/cyclone_pred
!git checkout feat/real-training-data
```

**No repo access at all** — zip the project locally, upload it as a Kaggle Dataset, attach it,
then copy it into the writable area:

```python
!cp -r /kaggle/input/<your-dataset-slug>/cyclone_pred /kaggle/working/
%cd /kaggle/working/cyclone_pred
```

`%cd` matters: the trainer resolves `data/` and `checkpoints/` relative to the working
directory.

---

## Step 4 — Download the training data

```python
!python -m src.data_pipeline.download_ibtracs
```

~105 MB, about 30 seconds. Fails instantly if Internet is off.

---

## Step 5 — Verify the dataset

Cheap, and it catches a truncated download before you burn GPU time:

```python
!python -m src.data_pipeline.cyclone_dataset --inspect --basin bay_of_bengal
```

Expect 2,182 train / 389 val samples, a class distribution, and one storm's observed future
track. If samples are 0, or the track deltas are all zeros, stop and re-download.

---

## Step 6 — Smoke test (~2 min)

```python
!python -m src.training.train_regional \
    --basin bay_of_bengal --epochs 2 --batch-size 16 \
    --max-train-samples 64 --num-workers 2 \
    --device cuda --checkpoint-dir /kaggle/working/ckpt_smoke
```

You want a validation track error printed and a checkpoint written. The *value* is meaningless
at this size — you are only confirming the loop runs on GPU.

---

## Step 7 — Train

```python
!python -m src.training.train_regional \
    --all-basins --device cuda \
    --batch-size 64 --num-workers 2 \
    --checkpoint-dir /kaggle/working/checkpoints
```

Notes on those flags:

- **`--checkpoint-dir /kaggle/working/checkpoints`** — required. The default `checkpoints/`
  is relative and would land outside the persisted directory if your `%cd` ever slips.
- **`--batch-size 64`** — a T4/P100 has 16 GB and the zero-imagery path is tiny. Drop to 16
  if you add `--imagery-root`, where the ResNet-50 becomes the memory bottleneck.
- **`--num-workers 2`** — Kaggle gives 4 vCPUs. Higher values oversubscribe and can hang
  notebook DataLoaders.

Roughly 30–45 min for all three basins at 60 epochs, dominated by `indian_ocean_south`
(19,093 samples).

### Using both T4s

The trainer is single-GPU by default. `--data-parallel` splits each batch across every
visible card:

```python
!python -m src.training.train_regional \
    --all-basins --device cuda --data-parallel \
    --batch-size 64 --num-workers 2 \
    --checkpoint-dir /kaggle/working/checkpoints
```

Expect ~1.5-1.7x, not 2x: DataParallel is one process with one optimiser that gathers every
output back to GPU 0 each step, and Kaggle's T4s talk over PCIe with no NVLink. Keep
`--batch-size` divisible by 2.

Honestly, at ~40 min single-GPU against a 30 h/week quota this is rarely worth it. The second
card earns its place once `--imagery-root` is in play and epochs run 5-8x longer.

---

## Step 8 — Keep the weights

**This is the step people forget.** A Kaggle session's files vanish when it ends.

```python
import json, glob, os

for path in sorted(glob.glob("/kaggle/working/checkpoints/*_history.json")):
    hist = json.load(open(path))
    best = min(hist, key=lambda r: r["val"].get("traj_km", 9e9))
    print(f"{os.path.basename(path):<36} "
          f"best epoch {best['epoch']:>3}  "
          f"track {best['val']['traj_km']:>7.1f} km  "
          f"acc {best['val']['accuracy']:.3f}  "
          f"wind MAE {best['val']['wind_mae']:.1f} kts")

!du -sh /kaggle/working/checkpoints
```

Then **either**:

- **Save Version** (top right) → *Save & Run All* — commits the notebook and everything in
  `/kaggle/working` as a versioned output you can download or attach to another notebook. This
  is the reliable path.
- Or download directly from the **Output** tab in the right panel.

Each basin checkpoint is ~400 MB (ResNet-50 + ConvLSTM). Kaggle's output cap is 20 GB, so
three basins are fine.

---

## Optional: satellite imagery

Everything above trains on best-track alone — the ResNet-50 receives zeros and contributes
nothing. That is an honest baseline, not a bug, but the visual branch is idle.

To use it, attach a HURSAT-B1 dataset (upload it yourself, or search Kaggle Datasets for
`HURSAT`) and point the trainer at it:

```python
!python -m src.training.train_regional \
    --basin bay_of_bengal --device cuda \
    --batch-size 16 --num-workers 2 \
    --imagery-root /kaggle/input/<hursat-dataset-slug> \
    --checkpoint-dir /kaggle/working/checkpoints
```

Two things change automatically: samples with no matching frame are dropped from the index,
and the ResNet-50 loads ImageNet pretrained weights (it stays randomly initialised in
zero-imagery mode, where pretraining would be pointless). Expect 5–8× longer per epoch.

---

## Troubleshooting

**`CUDA: False`** — Accelerator was not set, or the session started before you set it. Set it,
then **Run All** again; changing the accelerator restarts the session.

**`ModuleNotFoundError: No module named 'src'`** — you are not in the repo root. Re-run
`%cd /kaggle/working/cyclone_pred`. `%cd` persists across cells; `!cd` does not.

**`urllib`/connection error during download** — Internet toggle is off, or the account is not
phone-verified.

**`AttributeError: module 'torch.amp' has no attribute 'GradScaler'`** — Kaggle image has
torch < 2.4. Either use the older API by editing `train_regional.py`:

```python
self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)   # and torch.cuda.amp.autocast
```

or disable mixed precision, which costs speed but nothing else: set
`mixed_precision: false` in `configs/hyperparams.yaml`.

**DataLoader hangs at epoch start** — lower `--num-workers` to `0`. Notebook environments and
`persistent_workers` interact badly on some Kaggle images.

**Session died mid-run** — Kaggle caps interactive sessions at 12 h and idle at ~20 min (the
run itself counts as activity). This project needs well under an hour, so a death mid-run
almost always means the tab was closed. Use **Save & Run All** for unattended execution.

**Out of memory** — halve `--batch-size`. With imagery on, start at 16.

---

## Reading the results

The trainer prints per epoch and writes `checkpoints/<basin>_history.json`:

- **`track_km`** — mean great-circle error over the forecast horizon. The headline metric, and
  what checkpoint selection optimises.
- **`wind_mae`** — intensity error in knots.
- **`accuracy`** — IMD category accuracy. Always read it against the class distribution:
  `D`+`DD`+`CS` are ~80% of Bay of Bengal samples, so ~0.80 can mean the model never predicts a
  severe category at all.

For calibration: IMD's operational 24 h track error is roughly 80–100 km and its 24 h intensity
error roughly 10 kts. A first honest run will likely land well above both. Report what you
measure — the numbers in `history.json` are the only ones this project can currently stand
behind.
