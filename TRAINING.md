# Training Guide

How to train the regional cyclone models on real data, on a GPU machine.

---

## What changed

The old trainer learned from `torch.randn` images paired with `np.random.randint` labels —
input and target were statistically independent, so no amount of training could reduce the
loss below the entropy floor. That dataset is gone. Training now runs on **IBTrACS**, NOAA's
authoritative best-track archive: real storms, real winds, real observed motion.

| | Before | Now |
|---|---|---|
| Images | `torch.randn` | HURSAT-B1 IR, or zeros + mask |
| Labels | `np.random.randint(0, 7)` | IMD category from observed wind |
| Track target | `torch.randn(8, 4)` | Observed 6-hourly storm motion |
| Config | `hyperparams.yaml` ignored | Loaded; CLI overrides |
| Validation | none | Held out by season |
| Checkpoint | last epoch | Best val track error |
| Track loss | `SmoothL1` on raw lat/lon | Haversine, kilometres |
| Uncertainty | regressed on invented target | Gaussian NLL on real residuals |

---

## Step 0 — Machine

Any single NVIDIA GPU with **8 GB+** works. Reference points:

| GPU | Batch size | Approx. time, 60 epochs, BoB |
|---|---|---|
| T4 (Colab free) | 16 | ~40 min |
| RTX 3060 / 4060 | 24 | ~25 min |
| A100 / L4 | 48 | ~10 min |

These are for `--imagery none` (best-track only). With HURSAT imagery, multiply by roughly
5–8× — the ResNet-50 visual branch dominates.

---

## Step 1 — Environment

```bash
git clone <your repo> && cd cyclone_pred
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install the CUDA build of torch FIRST — requirements.txt would pull the CPU wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify the GPU is actually visible — if this prints `False`, everything below runs on CPU
at roughly 30× the cost:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 2 — Get the data

```bash
python -m src.data_pipeline.download_ibtracs
```

Pulls two CSVs (~105 MB total) into `data/raw/best_track/`:

- `ibtracs.NI.list.v04r01.csv` — North Indian → Bay of Bengal + Arabian Sea
- `ibtracs.SI.list.v04r01.csv` — South Indian

Already-downloaded files are skipped, so it is safe to re-run.

---

## Step 3 — Verify the dataset

Always run this before a long job. It is fast and catches a bad download immediately.

```bash
python -m src.data_pipeline.cyclone_dataset --inspect --basin bay_of_bengal
```

Expected (seasons 1980–2025, 6 held out for validation):

| Basin | Train | Val |
|---|---|---|
| `bay_of_bengal` | 2,182 | 389 |
| `arabian_sea` | 1,438 | 260 |
| `indian_ocean_south` | 19,093 | 3,199 |

The command also prints the IMD class distribution and the observed future track for one
sample. If the track deltas are all zeros, or sample counts are 0, stop and re-download.

---

## Step 4 — Smoke test (2 minutes)

Prove the loop runs end-to-end before committing to a full job:

```bash
python -m src.training.train_regional \
  --basin bay_of_bengal --epochs 2 --batch-size 8 \
  --max-train-samples 48 --num-workers 0 \
  --checkpoint-dir /tmp/ckpt_smoke
```

You want to see validation track error printed and a checkpoint written. The *value* is
meaningless at 2 epochs — you are only checking that nothing crashes.

---

## Step 5 — Train

**One basin:**

```bash
python -m src.training.train_regional \
  --basin bay_of_bengal --device cuda --batch-size 32
```

**All three, sequentially:**

```bash
python -m src.training.train_regional --all-basins --device cuda --batch-size 32
```

Epochs, LR, warmup, clipping, patience and loss weights all come from
`configs/hyperparams.yaml`. CLI flags override individual fields.

Per epoch you get:

```
[bay_of_bengal] epoch 12/60 (31.4s, lr=2.87e-04) | train loss 1.8241 acc 0.612 |
                val loss 2.0133 acc 0.574 track 118.3 km wind MAE 9.2 kts
```

Outputs land in `checkpoints/`:

- `<basin>_best.pth` — best **validation track error**, not the last epoch
- `<basin>_history.json` — full per-epoch metrics

Early stopping triggers after `early_stopping_patience` (default 10) epochs without
improvement.

---

## Step 6 — Add satellite imagery (optional, big lift)

Everything above trains on best-track alone. The visual branch receives zeros and
contributes nothing — a legitimate, honest baseline, but the ResNet-50 is idle.

To activate it, get **HURSAT-B1** (NOAA NCEI) — already cyclone-centered and keyed by
IBTrACS storm ID, which is why it needs no regridding:

```
https://www.ncei.noaa.gov/data/hursat-b1/
```

Extract the per-storm NetCDF files anywhere, then:

```bash
python -m src.data_pipeline.cyclone_dataset --inspect \
  --basin bay_of_bengal --imagery-root data/raw/hursat

python -m src.training.train_regional \
  --basin bay_of_bengal --device cuda --batch-size 16 \
  --imagery-root data/raw/hursat
```

Two things happen automatically when imagery is on: samples without a matching frame are
dropped from the index, and the ResNet-50 loads **ImageNet pretrained weights** (it stays
randomly initialised in zero-imagery mode, where pretraining would be pointless). Halve the
batch size — the visual branch is the memory bottleneck.

HURSAT-B1 carries one IR window channel, replicated across the 4-channel stack. A later
INSAT-3D ingest can supply true WV/VIS/TIR2 bands by implementing `load_frame` on a new
store class; nothing in the training code needs to change.

---

## Step 7 — Export for serving

```bash
python -m src.training.export_onnx --basin bay_of_bengal
```

Writes ONNX into `triton_model_repository/` for the Triton container in
`docker/docker-compose.yml`.

---

## Reading the metrics

- **`track_km`** — mean great-circle error across the forecast horizon. The headline number,
  and what checkpoint selection optimises.
- **`accuracy`** — IMD category accuracy. Read it alongside the class distribution: `D`+`DD`+`CS`
  are ~80% of Bay of Bengal samples, so ~0.80 can mean the model never predicts a severe
  category. The focal loss is class-weighted to fight this, but check per-class recall before
  claiming the model works.
- **`wind_mae`** — knots. IMD's own operational 24h intensity error is roughly 10 kts; beating
  that on held-out seasons would be a genuinely strong result.

For context, IMD's operational 24h track error is ~80–100 km. A first honest training run will
likely land well above that — say so rather than rounding it down.

---

## Known gaps

Things that are still not wired up. None of them block training; all of them affect how good
the result can get.

1. **ERA5 environment fields are not ingested.** Of the 8 synoptic inputs, only MSLP and the
   storm-motion proxy carry real values; SST, shear, vorticity and RH sit at their
   climatological means (z-scored to exactly 0.0 — no information, rather than misleading
   noise). `synoptic_mask` reports which slots are real. Wiring
   `fetch_copernicus_era5.py` into a `SynopticFeatureBuilder(era5_store=...)` is the
   single biggest remaining accuracy win, since shear and SST are the dominant physical
   drivers of intensification.

2. **`bay_of_bengal` and `arabian_sea` are small** (2.2k / 1.4k samples). A ResNet-50 has
   ~25M parameters and will overfit them quickly. Watch the train/val gap. Consider
   pretraining on `indian_ocean_south` (19k samples) and fine-tuning per basin.

3. **`SuCS` is ~0.3% of samples** (7 in Bay of Bengal training). The class weight is ~45×,
   which helps the loss but cannot manufacture examples. Treat any SuCS metric as noise.

4. **Inference still contains fabrication paths that will mask a bad model.**
   `MultimodalCycloneClassifier.predict_imd_intensity` substitutes a hardcoded
   wind/pressure table (plus `np.random.normal` jitter) whenever a head's output falls
   outside a physical range, and `CycloneTrajectoryConvLSTM.forecast_track_trajectory`
   overrides small deltas with "recurvature heuristics" and computes the cone radius from a
   fixed formula that ignores the model's own predicted uncertainty entirely. After training,
   these should be removed or put behind an explicit `trust_model=True` flag — otherwise a
   broken model still produces plausible-looking output, and you cannot tell the difference.

5. **`src/training/evaluate.py` still feeds `torch.randn`** as its satellite input and falls
   back to hardcoded constants (`42.5`, `84.2`, `5.8`) when a metric list is empty. Those
   constants are the "benchmark results" in the README. It needs rewriting against
   `RealCycloneDataset` before any reported number means anything.
