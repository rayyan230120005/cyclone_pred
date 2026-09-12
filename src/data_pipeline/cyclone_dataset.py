"""
Real Cyclone Training Dataset (IBTrACS best-track + optional HURSAT-B1 imagery).

Replaces the synthetic `torch.randn` dataset with genuine supervised pairs.

Ground truth comes from IBTrACS (NOAA NCEI), the authoritative best-track archive:
position, maximum sustained wind, central pressure and RMW at 3-6 hourly intervals
for every recorded storm since 1842.

Each emitted sample is one forecast instant `t` within one storm:

    satellite_image    [4, H, W]    imagery at time t         (HURSAT, or zero-filled)
    sequence_tensor    [8, 4, h, w] imagery at t-7..t         (history the ConvLSTM encodes)
    synoptic_vector    [8]          ERA5 / best-track derived environment at t
    class_label        scalar       IMD category index 0..6   (derived from wind)
    wind_speed         [1]          observed MSW, knots
    central_pressure   [1]          observed MSLP, hPa
    rmw                [1]          observed radius of max winds, km
    future_trajectory  [8, 4]       observed per-step [dlat, dlon, wind_kts, 0] for +6h..+48h
    history_motion     [8, 2]       observed per-step [dlat, dlon] for t-7..t
    last_motion        [2]          the most recent observed 6-hourly [dlat, dlon]

The trajectory target is the highest-value signal here and needs no imagery at all:
it is the storm's actual observed motion, which is exactly what the ConvLSTM decoder
is asked to predict.

`history_motion` / `last_motion` are the matching inputs. Past motion is the dominant
track predictor - pure persistence (continue the current heading) scores 180 km on Bay
of Bengal validation against 207 km for basin climatology - so a decoder that cannot
see heading cannot beat a baseline with no parameters at all.

Data tiers (the dataset is honest about what it has):
  - `imagery="hursat"`   real storm-centered satellite imagery; samples without it are dropped
  - `imagery="none"`     zero imagery + `has_imagery=0` mask, so the trainer can train a
                         legitimate best-track-only baseline instead of learning from noise

Usage:
    python -m src.data_pipeline.cyclone_dataset --inspect
    python -m src.data_pipeline.cyclone_dataset --inspect --basin arabian_sea
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_pipeline.preprocessor import MultimodalPreprocessor

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# IMD intensity scale (knots). Matches MultimodalCycloneClassifier.IMD_CLASSES.
# ----------------------------------------------------------------------------
IMD_CLASSES: List[str] = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]

# Lower bound (inclusive) of each category in knots.
IMD_WIND_LOWER_BOUNDS: List[float] = [17.0, 28.0, 34.0, 48.0, 64.0, 90.0, 120.0]


def imd_category_index(wind_kts: float) -> int:
    """
    Maps maximum sustained wind (knots) to an IMD category index 0..6.
    Winds below depression strength (17 kts) clamp to index 0.
    """
    idx = 0
    for i, lower in enumerate(IMD_WIND_LOWER_BOUNDS):
        if wind_kts >= lower:
            idx = i
    return idx


# ----------------------------------------------------------------------------
# Basin geography - mirrors configs/regional_bounds.json and ModelDispatcher
# ----------------------------------------------------------------------------
BASIN_BOUNDS: Dict[str, Dict[str, float]] = {
    "bay_of_bengal":      {"min_lat": 5.0,   "max_lat": 26.0, "min_lon": 80.0, "max_lon": 100.0},
    "arabian_sea":        {"min_lat": 5.0,   "max_lat": 26.0, "min_lon": 50.0, "max_lon": 79.99},
    "indian_ocean_south": {"min_lat": -35.0, "max_lat": 0.0,  "min_lon": 30.0, "max_lon": 120.0},
}

# Which IBTrACS basin file feeds which engine
BASIN_SOURCE_FILES: Dict[str, List[str]] = {
    "bay_of_bengal":      ["ibtracs.NI.list.v04r01.csv"],
    "arabian_sea":        ["ibtracs.NI.list.v04r01.csv"],
    "indian_ocean_south": ["ibtracs.SI.list.v04r01.csv"],
}


@dataclass
class SampleIndex:
    """
    One trainable instant.

    sid / name  identify the storm
    pos         row position of time `t` within that storm's track
    row_start   absolute row of the storm's first observation in the frame
    n_history   real history frames available at `pos` (<= history_steps; padded if short)
    n_forecast  real future steps available after `pos` (<= forecast_steps; masked if short)
    """
    sid: str
    name: str
    pos: int
    row_start: int
    n_history: int
    n_forecast: int


# ----------------------------------------------------------------------------
# Best-track loading
# ----------------------------------------------------------------------------
class BestTrackLoader:
    """
    Parses IBTrACS CSVs into a clean, 6-hourly, basin-filtered DataFrame.
    """

    # IBTrACS columns we consume. WMO_* is the agency-blended official record;
    # USA_* is the JTWC record, used as fallback where WMO is absent.
    USECOLS = [
        "SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON",
        "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES", "USA_RMW",
        "STORM_SPEED", "STORM_DIR", "DIST2LAND",
    ]

    def __init__(self, best_track_dir: str = os.path.join("data", "raw", "best_track")):
        self.best_track_dir = best_track_dir

    def _read_csv(self, path: str) -> pd.DataFrame:
        # IBTrACS row 0 is the header, row 1 is a units row ("", "degrees_north", ...).
        return pd.read_csv(
            path,
            skiprows=[1],
            usecols=lambda c: c in self.USECOLS,
            keep_default_na=False,
            na_values=["", " ", "NOT_NAMED", "-999", "-1"],
            low_memory=False,
        )

    def load_basin(self, basin: str, min_season: int = 1980) -> pd.DataFrame:
        """
        Loads and cleans every storm observation belonging to `basin`.
        """
        if basin not in BASIN_BOUNDS:
            raise ValueError(f"Unknown basin '{basin}'. Expected one of {list(BASIN_BOUNDS)}")

        frames = []
        for filename in BASIN_SOURCE_FILES[basin]:
            path = os.path.join(self.best_track_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing IBTrACS file: {path}\n"
                    f"Run: python -m src.data_pipeline.download_ibtracs"
                )
            frames.append(self._read_csv(path))

        df = pd.concat(frames, ignore_index=True)

        # --- type coercion ---
        for col in ["LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES",
                    "USA_RMW", "STORM_SPEED", "STORM_DIR", "DIST2LAND"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)

        df = df.dropna(subset=["ISO_TIME", "LAT", "LON"])
        df = df[df["SEASON"] >= min_season]

        # --- prefer the WMO blended record, fall back to JTWC ---
        df["wind_kts"] = df["WMO_WIND"].fillna(df["USA_WIND"])
        df["pres_hpa"] = df["WMO_PRES"].fillna(df["USA_PRES"])
        df["rmw_km"] = df["USA_RMW"] * 1.852  # IBTrACS stores RMW in nautical miles

        # An observation is only usable as an intensity label if wind is recorded.
        df = df.dropna(subset=["wind_kts"])
        df = df[df["wind_kts"] >= 17.0]  # at least depression strength on the IMD scale

        # --- restrict to synoptic hours (00/06/12/18 UTC) so steps are a clean 6h apart ---
        df = df[df["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])]
        df = df[df["ISO_TIME"].dt.minute == 0]

        # --- geographic filter for this regional engine ---
        b = BASIN_BOUNDS[basin]
        df = df[
            (df["LAT"] >= b["min_lat"]) & (df["LAT"] <= b["max_lat"]) &
            (df["LON"] >= b["min_lon"]) & (df["LON"] <= b["max_lon"])
        ]

        df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

        # Fill pressure where missing using the basin wind-pressure relationship,
        # then flag it so the pressure loss can be masked on estimated rows.
        df["pres_observed"] = df["pres_hpa"].notna().astype(np.float32)
        df["pres_hpa"] = df["pres_hpa"].fillna(1010.0 - (df["wind_kts"] / 3.92) ** 1.44)

        df["rmw_observed"] = df["rmw_km"].notna().astype(np.float32)
        df["rmw_km"] = df["rmw_km"].fillna((60.0 - df["wind_kts"] * 0.25).clip(lower=15.0))

        df["category_idx"] = df["wind_kts"].apply(imd_category_index).astype(np.int64)

        logger.info(
            f"[{basin}] loaded {len(df)} synoptic observations across "
            f"{df['SID'].nunique()} storms "
            f"(seasons {int(df['SEASON'].min())}-{int(df['SEASON'].max())})"
        )
        return df


# ----------------------------------------------------------------------------
# Imagery
# ----------------------------------------------------------------------------
class HursatImageryStore:
    """
    Loads storm-centered satellite imagery from a directory of HURSAT-B1 NetCDF files.

    HURSAT-B1 (NOAA NCEI) is already cyclone-centered and time-matched to IBTrACS by
    storm id, which is why it pairs with this dataset with no regridding.

    Files are looked up by IBTrACS SID. Expected layout:
        <root>/<SID>/*.nc      or      <root>/*<SID>*.nc

    HURSAT-B1 carries a single IR window channel (IRWIN). The 4-channel stack this
    project's models expect is formed by replicating it, with `has_imagery` set, so a
    later INSAT-3D ingest can supply true WV/VIS/TIR2 bands without changing this code.
    """

    def __init__(self, root: str, image_size: int = 256):
        self.root = root
        self.image_size = image_size
        self._index: Optional[Dict[str, List[str]]] = None

    def _build_index(self) -> Dict[str, List[str]]:
        if self._index is not None:
            return self._index
        index: Dict[str, List[str]] = {}
        if not self.root or not os.path.isdir(self.root):
            self._index = index
            return index
        for path in glob.glob(os.path.join(self.root, "**", "*.nc"), recursive=True):
            base = os.path.basename(path)
            parent = os.path.basename(os.path.dirname(path))
            for key in (parent, base):
                # IBTrACS SIDs look like 2020128N10086 - 13 chars, digits + hemisphere letter
                for token in key.replace(".", "_").split("_"):
                    if len(token) == 13 and token[8] in "NS" and token[:4].isdigit():
                        index.setdefault(token, []).append(path)
        self._index = index
        logger.info(f"HURSAT index built: {len(index)} storms under {self.root}")
        return index

    def has_storm(self, sid: str) -> bool:
        return sid in self._build_index()

    def load_frame(self, sid: str, timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """
        Returns a [4, H, W] brightness-temperature array in Kelvin, or None if absent.
        """
        paths = self._build_index().get(sid)
        if not paths:
            return None

        try:
            import xarray as xr
        except ImportError:
            logger.warning("xarray not installed; imagery disabled. pip install xarray netCDF4")
            return None

        # HURSAT filenames embed the observation time; pick the matching file.
        stamp = timestamp.strftime("%Y%m%d%H")
        best = next((p for p in paths if stamp in os.path.basename(p)), None)
        if best is None:
            return None

        try:
            with xr.open_dataset(best) as ds:
                var = next((v for v in ("IRWIN", "irwin", "ch4", "IRWIN_2") if v in ds), None)
                if var is None:
                    return None
                arr = np.asarray(ds[var].values, dtype=np.float32)
        except Exception as exc:  # corrupt or truncated download
            logger.debug(f"Failed to read {best}: {exc}")
            return None

        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None

        arr = self._center_crop_resize(arr, self.image_size)
        # Replicate the IR window band across the 4-channel stack the models expect.
        return np.stack([arr, arr, arr, arr], axis=0)

    @staticmethod
    def _center_crop_resize(arr: np.ndarray, size: int) -> np.ndarray:
        fill = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 280.0
        t = torch.from_numpy(arr)[None, None]
        t = torch.nan_to_num(t, nan=fill)
        t = torch.nn.functional.interpolate(
            t, size=(size, size), mode="bilinear", align_corners=False
        )
        return t[0, 0].numpy()


# ----------------------------------------------------------------------------
# Environment (synoptic) features
# ----------------------------------------------------------------------------
class SynopticFeatureBuilder:
    """
    Assembles the 8-element environment vector the Dense branch consumes:
        [sst, shear, vorticity_850, rh_700, mslp, u_850, v_850, sst_anomaly]

    ERA5 supplies the first four and the winds. Where an ERA5 cache is absent, the
    features that best-track genuinely provides (MSLP, and storm motion as a proxy
    for the steering flow) are filled from IBTrACS and the rest are set to the
    climatological mean - which the preprocessor's z-score maps to exactly 0.0,
    i.e. "no information", rather than to misleading noise.

    `synoptic_mask` reports which of the 8 slots are real observations.
    """

    def __init__(self, era5_store: Optional[Any] = None):
        self.era5_store = era5_store
        self.means = MultimodalPreprocessor.SYNOPTIC_MEANS

    def build(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        vec = self.means.copy()
        mask = np.zeros(8, dtype=np.float32)

        # --- slot 4: mean sea level pressure (observed or wind-pressure estimated) ---
        vec[4] = float(row["pres_hpa"])
        mask[4] = float(row.get("pres_observed", 0.0))

        # --- slots 5,6: storm motion vector as steering-flow proxy ---
        speed = row.get("STORM_SPEED", np.nan)
        direction = row.get("STORM_DIR", np.nan)
        if pd.notna(speed) and pd.notna(direction):
            theta = np.deg2rad(float(direction))
            # IBTrACS STORM_DIR is the heading the storm moves toward,
            # in degrees clockwise from north.
            vec[5] = float(speed) * np.sin(theta)   # eastward component, knots
            vec[6] = float(speed) * np.cos(theta)   # northward component, knots
            mask[5] = 1.0
            mask[6] = 1.0

        # --- ERA5 fields, when a cache is wired in ---
        if self.era5_store is not None:
            era5 = self.era5_store.lookup(float(row["LAT"]), float(row["LON"]), row["ISO_TIME"])
            if era5:
                for slot, key in [
                    (0, "sst_celsius"),
                    (1, "vertical_wind_shear_kts"),
                    (2, "relative_vorticity_850_s1"),
                    (3, "relative_humidity_700_pct"),
                    (7, "sst_anomaly_celsius"),
                ]:
                    if era5.get(key) is not None:
                        vec[slot] = float(era5[key])
                        mask[slot] = 1.0

        return vec.astype(np.float32), mask


# ----------------------------------------------------------------------------
# The Dataset
# ----------------------------------------------------------------------------
class RealCycloneDataset(Dataset):
    """
    Supervised cyclone samples built from IBTrACS best-track, with optional imagery.

    Args:
        basin:           one of BASIN_BOUNDS
        split:           "train" | "val" - split is by SEASON so no storm leaks across
        val_seasons:     number of most recent seasons held out for validation
        history_steps:   frames of history the ConvLSTM encodes (default 8 = 48h)
        forecast_steps:  steps to predict (default 8 = +6h..+48h)
        imagery:         "hursat" (requires imagery_root) or "none"
        augment:         random rotation/flip/noise on the imagery
    """

    def __init__(
        self,
        basin: str = "bay_of_bengal",
        split: str = "train",
        best_track_dir: str = os.path.join("data", "raw", "best_track"),
        imagery: str = "none",
        imagery_root: Optional[str] = None,
        era5_store: Optional[Any] = None,
        min_season: int = 1980,
        val_seasons: int = 6,
        history_steps: int = 8,
        forecast_steps: int = 8,
        min_history: int = 2,
        min_forecast_steps: int = 4,
        image_size: int = 256,
        sequence_image_size: int = 128,
        augment: bool = True,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        if imagery not in ("hursat", "none"):
            raise ValueError(f"imagery must be 'hursat' or 'none', got '{imagery}'")

        self.basin = basin
        self.split = split
        self.imagery_mode = imagery
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.min_history = max(1, min(min_history, history_steps))
        self.min_forecast_steps = max(1, min(min_forecast_steps, forecast_steps))
        self.image_size = image_size
        self.sequence_image_size = sequence_image_size
        self.augment = augment and split == "train"

        self.preprocessor = MultimodalPreprocessor()
        self.synoptic_builder = SynopticFeatureBuilder(era5_store=era5_store)
        self.imagery_store = (
            HursatImageryStore(imagery_root, image_size=image_size)
            if imagery == "hursat" and imagery_root else None
        )

        # --- load and split by season ---
        df = BestTrackLoader(best_track_dir).load_basin(basin, min_season=min_season)
        seasons = sorted(df["SEASON"].unique())
        if len(seasons) <= val_seasons:
            raise ValueError(
                f"[{basin}] only {len(seasons)} seasons available; "
                f"cannot hold out {val_seasons}."
            )
        cutoff = seasons[-val_seasons]
        df = df[df["SEASON"] < cutoff] if split == "train" else df[df["SEASON"] >= cutoff]
        self.df = df.reset_index(drop=True)

        self.samples = self._build_sample_index()
        if not self.samples:
            raise ValueError(
                f"[{basin}/{split}] produced 0 samples. "
                f"Check the best-track download and basin bounds."
            )

        logger.info(
            f"[{basin}/{split}] {len(self.samples)} samples from "
            f"{self.df['SID'].nunique()} storms | imagery={imagery}"
        )

    # -- index construction ---------------------------------------------------
    def _build_sample_index(self) -> List[SampleIndex]:
        """
        Builds every trainable instant in the split.

        Demanding a full 8 frames of history *and* 8 future steps would discard most of
        the archive - the median North Indian storm is only ~12 synoptic observations
        long, shorter than that 15-step window. Instead:

          - history shorter than `history_steps` is left-padded by repeating the
            earliest real frame, the standard treatment for a warm-up sequence;
          - a forecast shorter than `forecast_steps` is kept and the missing tail is
            masked out of the loss, so a storm that dissipates at +30h still teaches
            the model everything it observed up to +30h.

        Only `min_history` real frames and `min_forecast_steps` real future steps are
        actually required. Steps must still sit on a contiguous 6-hourly grid - a gap
        truncates the run rather than silently spanning it.
        """
        samples: List[SampleIndex] = []

        for sid, group in self.df.groupby("SID", sort=False):
            row_start = int(group.index[0])
            times = group["ISO_TIME"].to_numpy()
            n = len(group)
            if n < self.min_history + self.min_forecast_steps:
                continue
            if self.imagery_store is not None and not self.imagery_store.has_storm(sid):
                continue

            # step_ok[i] is True when observation i follows i-1 by exactly 6 hours.
            step_ok = np.zeros(n, dtype=bool)
            step_ok[1:] = np.diff(times) == np.timedelta64(6, "h")

            name = str(group["NAME"].iloc[0])
            for pos in range(n):
                # How far back a contiguous run extends from `pos`.
                n_history = 1
                while (
                    n_history < self.history_steps
                    and pos - n_history + 1 > 0
                    and step_ok[pos - n_history + 1]
                ):
                    n_history += 1
                if n_history < self.min_history:
                    continue

                # How far forward a contiguous run extends from `pos`.
                n_forecast = 0
                while (
                    n_forecast < self.forecast_steps
                    and pos + n_forecast + 1 < n
                    and step_ok[pos + n_forecast + 1]
                ):
                    n_forecast += 1
                if n_forecast < self.min_forecast_steps:
                    continue

                samples.append(
                    SampleIndex(
                        sid=sid,
                        name=name,
                        pos=pos,
                        row_start=row_start,
                        n_history=n_history,
                        n_forecast=n_forecast,
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # -- item construction ----------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        curr = self.df.iloc[s.row_start + s.pos]

        # ---------- imagery ----------
        image, sequence, has_imagery = self._load_imagery(s, curr)

        # ---------- environment ----------
        syn_vec, syn_mask = self.synoptic_builder.build(curr)
        synoptic = self.preprocessor.normalize_synoptic_vector(syn_vec)

        # ---------- intensity targets (observed) ----------
        wind = torch.tensor([float(curr["wind_kts"])], dtype=torch.float32)
        pressure = torch.tensor([float(curr["pres_hpa"])], dtype=torch.float32)
        rmw = torch.tensor([float(curr["rmw_km"])], dtype=torch.float32)
        label = torch.tensor(int(curr["category_idx"]), dtype=torch.long)

        # ---------- past motion (the dominant track predictor) ----------
        history_motion, last_motion = self._build_history_motion(s)

        # ---------- trajectory target (observed future motion) ----------
        future, traj_mask = self._build_future_trajectory(s)

        return {
            "satellite_image": image,
            "sequence_tensor": sequence,
            "synoptic_vector": synoptic,
            "synoptic_mask": torch.from_numpy(syn_mask),
            "class_label": label,
            "wind_speed": wind,
            "central_pressure": pressure,
            "rmw": rmw,
            "future_trajectory": future,
            "trajectory_mask": traj_mask,
            "history_motion": history_motion,
            "last_motion": last_motion,
            "has_imagery": torch.tensor(has_imagery, dtype=torch.float32),
            "pressure_observed": torch.tensor(
                float(curr.get("pres_observed", 0.0)), dtype=torch.float32
            ),
            "rmw_observed": torch.tensor(
                float(curr.get("rmw_observed", 0.0)), dtype=torch.float32
            ),
            "current_lat": torch.tensor(float(curr["LAT"]), dtype=torch.float32),
            "current_lon": torch.tensor(float(curr["LON"]), dtype=torch.float32),
        }

    def _load_imagery(
        self, s: SampleIndex, curr: pd.Series
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns (image [4,H,W], sequence [T,4,h,w], has_imagery flag).

        With imagery disabled the tensors are exact zeros, never random noise: a zero
        input contributes no gradient signal to the visual branch, whereas noise would
        teach the network to read meaning out of nothing.
        """
        img_size = self.image_size
        seq_size = self.sequence_image_size
        blank = (
            torch.zeros(4, img_size, img_size),
            torch.zeros(self.history_steps, 4, seq_size, seq_size),
            0.0,
        )

        if self.imagery_store is None:
            return blank

        frame = self.imagery_store.load_frame(s.sid, curr["ISO_TIME"])
        if frame is None:
            return blank

        image = self.preprocessor.normalize_satellite_tensor(frame)
        if self.augment:
            image = self.preprocessor.augment_sample(image)

        # History frames, oldest first. Where the storm has fewer than `history_steps`
        # real observations the run is left-padded by repeating its earliest frame,
        # so the ConvLSTM still receives a full-length warm-up sequence.
        frames = []
        for k in range(self.history_steps - 1, -1, -1):
            offset = min(k, s.n_history - 1)
            hist_row = self.df.iloc[s.row_start + s.pos - offset]
            hist = self.imagery_store.load_frame(s.sid, hist_row["ISO_TIME"])
            frames.append(
                self.preprocessor.normalize_satellite_tensor(hist if hist is not None else frame)
            )

        sequence = torch.nn.functional.interpolate(
            torch.stack(frames, dim=0),
            size=(seq_size, seq_size),
            mode="bilinear",
            align_corners=False,
        )
        return image, sequence, 1.0

    def _build_history_motion(self, s: SampleIndex) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The storm's observed motion over the preceding frames.

        Returns:
            history_motion [history_steps, 2]  per-step [dlat, dlon], oldest first,
                                               left-padded with zeros where the storm
                                               has less history than the window
            last_motion    [2]                 the most recent 6-hourly [dlat, dlon]

        This is the single most predictive track feature there is: simple persistence
        (continue the current heading) beats basin climatology by a wide margin. The
        decoder previously received neither, so it could not represent heading at all
        and was structurally incapable of beating a persistence baseline.
        """
        base = s.row_start + s.pos
        motion = np.zeros((self.history_steps, 2), dtype=np.float32)

        # Walk backwards over the real history, filling the window from the right.
        for k in range(1, s.n_history):
            curr = self.df.iloc[base - k + 1]
            prev = self.df.iloc[base - k]
            dlon = float(curr["LON"]) - float(prev["LON"])
            if dlon > 180.0:
                dlon -= 360.0
            elif dlon < -180.0:
                dlon += 360.0
            motion[self.history_steps - k] = [
                float(curr["LAT"]) - float(prev["LAT"]),
                dlon,
            ]

        last = motion[-1].copy()
        return torch.from_numpy(motion), torch.from_numpy(last)

    def _build_future_trajectory(self, s: SampleIndex) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Observed per-step motion for +6h..+48h as [dlat, dlon, wind_kts, 0.0],
        plus a [forecast_steps] mask marking which steps were actually observed.

        Deltas are incremental (step-to-step), matching how
        CycloneTrajectoryConvLSTM.forecast_track_trajectory accumulates them.
        Slot 3 is left at zero: track-spread uncertainty is not directly observable,
        so the trainer learns it via Gaussian NLL against the model's own residuals
        rather than regressing a fabricated target.
        """
        base = s.row_start + s.pos
        steps = np.zeros((self.forecast_steps, 4), dtype=np.float32)
        mask = np.zeros(self.forecast_steps, dtype=np.float32)

        prev_lat = float(self.df.iloc[base]["LAT"])
        prev_lon = float(self.df.iloc[base]["LON"])

        for k in range(1, s.n_forecast + 1):
            nxt = self.df.iloc[base + k]
            lat, lon = float(nxt["LAT"]), float(nxt["LON"])
            dlon = lon - prev_lon
            # Guard the antimeridian so a wrap never becomes a 360-degree jump.
            if dlon > 180.0:
                dlon -= 360.0
            elif dlon < -180.0:
                dlon += 360.0
            steps[k - 1] = [lat - prev_lat, dlon, float(nxt["wind_kts"]), 0.0]
            mask[k - 1] = 1.0
            prev_lat, prev_lon = lat, lon

        return torch.from_numpy(steps), torch.from_numpy(mask)

    # -- reporting ------------------------------------------------------------
    def _category_counts(self) -> np.ndarray:
        counts = np.zeros(len(IMD_CLASSES), dtype=np.float64)
        for s in self.samples:
            counts[int(self.df.iloc[s.row_start + s.pos]["category_idx"])] += 1
        return counts

    def class_distribution(self) -> Dict[str, int]:
        counts = self._category_counts()
        return {code: int(counts[i]) for i, code in enumerate(IMD_CLASSES)}

    def class_weights(self, power: float = 0.5, max_weight: float = 8.0) -> torch.Tensor:
        """
        Class weights for the focal loss. Severe categories are rare; without any
        weighting the model can score well by never predicting ESCS or SuCS.

        Full inverse frequency is too violent here. Bay of Bengal has 7 SuCS samples
        against 839 depressions, giving that class a weight of ~45x - so a single
        SuCS sample in a batch produces a gradient spike 45 times the norm, which is
        what makes validation accuracy swing between 0.09 and 0.42 epoch to epoch.

        Weights are therefore raised to `power` (0.5 = inverse square root, the usual
        compromise) and clipped at `max_weight`. Rare classes stay up-weighted; no
        single class can hijack a batch. Set power=1.0, max_weight=inf to recover the
        original behaviour.
        """
        counts = np.maximum(self._category_counts(), 1.0)
        weights = (counts.sum() / (len(IMD_CLASSES) * counts)) ** power
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.mean()  # keep the loss scale comparable across basins
        return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------------------
def _inspect(basin: str, imagery_root: Optional[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mode = "hursat" if imagery_root else "none"
    train = RealCycloneDataset(
        basin=basin, split="train", imagery=mode, imagery_root=imagery_root
    )
    val = RealCycloneDataset(
        basin=basin, split="val", imagery=mode, imagery_root=imagery_root, augment=False
    )

    print(f"\n=== {basin} ===")
    print(f"train samples : {len(train)}")
    print(f"val samples   : {len(val)}")
    print(f"imagery mode  : {mode}")

    print("\nIMD class distribution (train):")
    for code, n in train.class_distribution().items():
        pct = 100.0 * n / max(len(train), 1)
        print(f"  {code:<5} {n:>6}  ({pct:4.1f}%)")

    print("\nFocal-loss class weights:")
    print(" ", np.round(train.class_weights().numpy(), 3).tolist())

    sample = train[0]
    print("\nSample tensor shapes:")
    for key, tensor in sample.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key:<20} {str(tuple(tensor.shape)):<20} {tensor.dtype}")

    mask_row = sample["trajectory_mask"].tolist()
    print("\nObserved future track (+6h..+48h)  [dlat, dlon, wind_kts]:")
    for i, step in enumerate(sample["future_trajectory"].tolist()):
        flag = "" if mask_row[i] else "   (unobserved, masked out of loss)"
        print(f"  +{(i + 1) * 6:>2}h  dlat={step[0]:+.2f}  dlon={step[1]:+.2f}  wind={step[2]:.0f} kts{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the real cyclone dataset.")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--basin", default="bay_of_bengal", choices=list(BASIN_BOUNDS))
    parser.add_argument("--imagery-root", default=None, help="Directory of HURSAT-B1 NetCDF files")
    args = parser.parse_args()

    if args.inspect:
        _inspect(args.basin, args.imagery_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
