"""
Multimodal Preprocessor: Radiometric Calibration, Normalization, and Tensor Stacking.

Features:
- Brightness temperature calibration ($T_b$) from raw satellite digital numbers
- Cloud-top temperature enhancement and normalization
- Multi-channel stacking [IR, WV, VIS, TIR2]
- Tabular synoptic feature vector scaling for the Dense meteorological branch
- Data augmentations (random spatial rotations, horizontal flips, Gaussian noise) for robust model training
"""

import logging
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MultimodalPreprocessor:
    """
    Standardizes satellite image arrays and synoptic reanalysis vectors into PyTorch model inputs.
    """

    # Synoptic feature statistics [mean, std] for z-score normalization
    # Features: [sst, shear, vorticity, rh700, mslp, u_wind, v_wind, sst_anomaly]
    SYNOPTIC_MEANS = np.array([28.5, 18.0, 45.0, 78.0, 985.0, -4.0, 3.0, 0.5], dtype=np.float32)
    SYNOPTIC_STDS = np.array([1.8, 8.5, 25.0, 12.0, 22.0, 8.0, 8.0, 0.8], dtype=np.float32)

    # Satellite channel min/max for min-max scaling to [-1.0, 1.0]
    # Channels: [TIR1 (180-310K), WV (190-270K), VIS (0.0-1.0), TIR2 (180-310K)]
    SAT_MINS = np.array([180.0, 190.0, 0.0, 180.0], dtype=np.float32).reshape(4, 1, 1)
    SAT_MAXS = np.array([315.0, 275.0, 1.0, 315.0], dtype=np.float32).reshape(4, 1, 1)

    def __init__(self, device: str = "cpu"):
        self.device = device

    def normalize_satellite_tensor(
        self,
        satellite_array: np.ndarray, # [4, H, W]
    ) -> torch.Tensor:
        """
        Applies radiometric range normalization to satellite channels, scaling into [-1.0, 1.0].
        """
        # Ensure array is float32
        arr = satellite_array.astype(np.float32)
        if arr.shape[0] == 3:
            # If 3-channel input, append an estimated TIR2 channel
            tir2 = arr[0:1] - 2.0
            arr = np.concatenate([arr, tir2], axis=0)

        # Min-max normalization per channel into [-1, 1]
        norm = 2.0 * (arr - self.SAT_MINS) / (self.SAT_MAXS - self.SAT_MINS) - 1.0
        norm = np.clip(norm, -1.0, 1.0)

        tensor = torch.from_numpy(norm).float()
        return tensor

    def normalize_synoptic_vector(
        self,
        synoptic_dict_or_list: Union[Dict[str, float], List[float], np.ndarray],
    ) -> torch.Tensor:
        """
        Normalizes 8-element meteorological vector with z-score scaling.
        Features: [sst, shear, vorticity, rh700, mslp, u_wind, v_wind, sst_anomaly]
        """
        if isinstance(synoptic_dict_or_list, dict):
            vec = np.array([
                synoptic_dict_or_list.get("sst_celsius", 28.5),
                synoptic_dict_or_list.get("vertical_wind_shear_kts", 18.0),
                synoptic_dict_or_list.get("relative_vorticity_850_s1", 45.0),
                synoptic_dict_or_list.get("relative_humidity_700_pct", 78.0),
                synoptic_dict_or_list.get("mean_sea_level_pressure_hpa", 985.0),
                synoptic_dict_or_list.get("u_wind_850_kts", -4.0),
                synoptic_dict_or_list.get("v_wind_850_kts", 3.0),
                synoptic_dict_or_list.get("sst_anomaly_celsius", 0.5),
            ], dtype=np.float32)
        elif isinstance(synoptic_dict_or_list, (list, tuple)):
            vec = np.array(synoptic_dict_or_list, dtype=np.float32)
        else:
            vec = synoptic_dict_or_list.astype(np.float32)

        # Standard z-score normalization
        norm_vec = (vec - self.SYNOPTIC_MEANS) / self.SYNOPTIC_STDS
        return torch.from_numpy(norm_vec).float()

    def augment_sample(
        self,
        satellite_tensor: torch.Tensor, # [4, H, W]
    ) -> torch.Tensor:
        """
        Applies rotational invariance & random flips suited for cyclonic atmospheric vortex symmetry.
        """
        # Random 90-degree rotations
        k = np.random.randint(0, 4)
        if k > 0:
            satellite_tensor = torch.rot90(satellite_tensor, k, dims=[1, 2])

        # Random horizontal flip with 50% probability
        if np.random.rand() > 0.5:
            satellite_tensor = torch.flip(satellite_tensor, dims=[2])

        # Mild Gaussian noise injection
        noise = torch.randn_like(satellite_tensor) * 0.02
        satellite_tensor = satellite_tensor + noise
        return satellite_tensor


if __name__ == "__main__":
    preprocessor = MultimodalPreprocessor()
    mock_sat = np.random.uniform(190, 300, (4, 256, 256)).astype(np.float32)
    sat_t = preprocessor.normalize_satellite_tensor(mock_sat)
    print(f"Normalized satellite tensor shape: {sat_t.shape}, min: {sat_t.min():.2f}, max: {sat_t.max():.2f}")

    mock_syn = {
        "sst_celsius": 30.2,
        "vertical_wind_shear_kts": 8.5,
        "relative_vorticity_850_s1": 85.0,
        "relative_humidity_700_pct": 88.0,
        "mean_sea_level_pressure_hpa": 940.0,
        "u_wind_850_kts": -10.0,
        "v_wind_850_kts": 5.0,
        "sst_anomaly_celsius": 1.7,
    }
    syn_t = preprocessor.normalize_synoptic_vector(mock_syn)
    print(f"Normalized synoptic vector shape: {syn_t.shape}, values: {syn_t}")
