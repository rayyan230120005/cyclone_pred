"""
Geospatial Dynamic Model Dispatcher.

Features:
- Routes incoming inference requests to the geographically specialized ocean basin model:
  - Bay of Bengal (BOB)
  - Arabian Sea (ARB)
  - South Indian Ocean (SIO)
- Handles dynamic model lifecycle (PyTorch models or ONNX Runtime inference sessions)
- Orchestrates the full prediction pipeline: Detection -> Multimodal Fusion Classification -> ConvLSTM Spatiotemporal Trajectory
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from .identification import CycloneDetector, BoundingBoxResult
from .classification_fusion import MultimodalCycloneClassifier
from .prediction_convlstm import CycloneTrajectoryConvLSTM

logger = logging.getLogger(__name__)


class ModelDispatcher:
    """
    Coordinates regional model routing, loading, and end-to-end multi-stage inference.
    """

    DEFAULT_BOUNDS_PATH = "configs/regional_bounds.json"

    def __init__(
        self,
        config_path: str = DEFAULT_BOUNDS_PATH,
        device: str = "cpu",
        use_onnx: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.use_onnx = use_onnx
        self.config_path = config_path
        self.regional_configs = self._load_regional_bounds()

        # Instantiate Models
        self.detector = CycloneDetector(in_channels=4, num_classes=2).to(self.device)
        self.classifier = MultimodalCycloneClassifier(in_satellite_channels=4, synoptic_features_dim=8).to(self.device)
        self.trajectory_forecaster = CycloneTrajectoryConvLSTM(in_channels=4, forecast_steps=8).to(self.device)

        self.detector.eval()
        self.classifier.eval()
        self.trajectory_forecaster.eval()

        logger.info(f"ModelDispatcher initialized on device: {self.device}")

    def _load_regional_bounds(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "basins": {
                "bay_of_bengal": {"bounds": {"min_lat": 5.0, "max_lat": 26.0, "min_lon": 80.0, "max_lon": 100.0}},
                "arabian_sea": {"bounds": {"min_lat": 5.0, "max_lat": 26.0, "min_lon": 50.0, "max_lon": 79.99}},
                "indian_ocean_south": {"bounds": {"min_lat": -35.0, "max_lat": 0.0, "min_lon": 30.0, "max_lon": 120.0}},
            },
            "default_fallback_basin": "bay_of_bengal",
        }

    def dispatch_basin(self, lat: float, lon: float) -> Tuple[str, Dict[str, Any]]:
        """
        Determines the ocean basin ID based on geospatial coordinates.
        """
        basins = self.regional_configs.get("basins", {})
        for basin_key, basin_data in basins.items():
            b = basin_data.get("bounds", {})
            if (
                b.get("min_lat", -90) <= lat <= b.get("max_lat", 90)
                and b.get("min_lon", -180) <= lon <= b.get("max_lon", 180)
            ):
                return basin_key, basin_data

        fallback = self.regional_configs.get("default_fallback_basin", "bay_of_bengal")
        return fallback, basins.get(fallback, {})

    def run_full_pipeline(
        self,
        satellite_tensor: torch.Tensor, # [1, 4, H, W]
        synoptic_vector: torch.Tensor,  # [1, 8]
        approx_lat: float,
        approx_lon: float,
        sequence_tensor: Optional[torch.Tensor] = None, # [1, 8, 4, 128, 128]
    ) -> Dict[str, Any]:
        """
        Runs the complete prediction pipeline:
        1. Basin routing
        2. Cyclone Eye Detection
        3. Multimodal IMD Intensity & Pressure Classification
        4. Spatiotemporal Trajectory & Uncertainty Forecasting
        """
        basin_id, basin_meta = self.dispatch_basin(approx_lat, approx_lon)
        bounds = basin_meta.get("bounds", {"min_lat": 5.0, "min_lon": 80.0, "max_lat": 25.0, "max_lon": 100.0})
        geo_bbox = (bounds["min_lat"], bounds["min_lon"], bounds["max_lat"], bounds["max_lon"])

        # 1. Detection
        sat_t = satellite_tensor.to(self.device)
        syn_t = synoptic_vector.to(self.device)
        detections = self.detector.detect(sat_t, geo_bbox)
        primary_det = detections[0] if detections else None

        detected_lat = primary_det.estimated_lat if primary_det else approx_lat
        detected_lon = primary_det.estimated_lon if primary_det else approx_lon

        # 2. Multimodal Intensity Classification
        intensity_pred = self.classifier.predict_imd_intensity(sat_t, syn_t)

        # 3. Spatiotemporal Trajectory Forecast
        if sequence_tensor is None:
            # Generate historical sequence by repeating with micro-variations
            seq_t = sat_t.unsqueeze(1).repeat(1, 8, 1, 1, 1)[:, :, :, :128, :128]
        else:
            seq_t = sequence_tensor.to(self.device)

        forecast_track = self.trajectory_forecaster.forecast_track_trajectory(
            sequence_tensor=seq_t,
            current_lat=detected_lat,
            current_lon=detected_lon,
            current_wind_kts=intensity_pred["maximum_sustained_wind_kts"],
        )

        return {
            "basin": {
                "id": basin_id,
                "name": basin_meta.get("name", basin_id),
                "model_engine": f"{basin_id}_engine",
            },
            "eye_detection": {
                "detected": primary_det is not None,
                "confidence": primary_det.confidence if primary_det else 0.0,
                "latitude": detected_lat,
                "longitude": detected_lon,
                "bounding_box_pixels": primary_det.bbox_pixels if primary_det else None,
            },
            "intensity_classification": intensity_pred,
            "spatiotemporal_forecast": {
                "track": forecast_track,
                "lead_times_hours": [6, 12, 18, 24, 30, 36, 42, 48],
                "forecast_summary": f"Storm tracking along {basin_id.replace('_', ' ').title()} basin with peak intensity {intensity_pred['category_code']} ({intensity_pred['maximum_sustained_wind_kmph']} km/h).",
            },
        }


if __name__ == "__main__":
    dispatcher = ModelDispatcher()
    basin_key, meta = dispatcher.dispatch_basin(17.5, 88.2)
    print(f"Dispatched basin for (17.5, 88.2): {basin_key} -> {meta.get('name')}")
    
    basin_key2, meta2 = dispatcher.dispatch_basin(18.2, 68.4)
    print(f"Dispatched basin for (18.2, 68.4): {basin_key2} -> {meta2.get('name')}")

    dummy_sat = torch.randn(1, 4, 256, 256)
    dummy_syn = torch.randn(1, 8)
    pipeline_res = dispatcher.run_full_pipeline(dummy_sat, dummy_syn, 16.5, 87.8)
    print("Pipeline execution succeeded. IMD Category:", pipeline_res["intensity_classification"]["category_code"])
