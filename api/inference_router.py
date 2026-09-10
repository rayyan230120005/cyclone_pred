"""
FastAPI Inference Router for Cyclone Prediction & Geospatial Telemetry.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from pydantic import BaseModel, Field
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form

from src.models.model_dispatcher import ModelDispatcher
from .alerts_whatsapp import broadcast_alert_for_prediction
from src.data_pipeline.fetch_mosdac_insat import MosdacInsatFetcher
from src.data_pipeline.fetch_copernicus_era5 import CopernicusEra5Fetcher
from src.data_pipeline.fetch_openmeteo import OpenMeteoFetcher
from src.data_pipeline.preprocessor import MultimodalPreprocessor
from src.utils.gis_visualization import create_leaflet_vector_payload
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Tropical Cyclone Inference"])

# Shared pipeline instances
_dispatcher: Optional[ModelDispatcher] = None
_preprocessor = MultimodalPreprocessor()
_mosdac_fetcher = MosdacInsatFetcher()
_era5_fetcher = CopernicusEra5Fetcher()
_openmeteo_fetcher = OpenMeteoFetcher()


def get_dispatcher() -> ModelDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ModelDispatcher()
    return _dispatcher


# ==========================================
# Pydantic Schemas
# ==========================================

class SynopticInput(BaseModel):
    sst_celsius: Optional[float] = Field(29.5, description="Sea Surface Temperature in °C")
    vertical_wind_shear_kts: Optional[float] = Field(12.0, description="850-200 hPa Vertical Wind Shear in knots")
    relative_vorticity_850_s1: Optional[float] = Field(65.0, description="850 hPa Relative Vorticity (x10^-5 s^-1)")
    relative_humidity_700_pct: Optional[float] = Field(85.0, description="700 hPa Relative Humidity in %")
    mean_sea_level_pressure_hpa: Optional[float] = Field(975.0, description="Mean Sea Level Pressure in hPa")
    u_wind_850_kts: Optional[float] = Field(-6.0, description="Zonal wind component in knots")
    v_wind_850_kts: Optional[float] = Field(4.0, description="Meridional wind component in knots")
    sst_anomaly_celsius: Optional[float] = Field(1.2, description="SST anomaly from climatological baseline in °C")


class InferenceRequest(BaseModel):
    latitude: float = Field(..., description="Estimated cyclone center latitude (-40.0 to 30.0)")
    longitude: float = Field(..., description="Estimated cyclone center longitude (30.0 to 120.0)")
    synoptic_features: Optional[SynopticInput] = Field(default_factory=SynopticInput)
    fetch_live_weather: Optional[bool] = Field(True, description="Fetch live marine weather from Open-Meteo")
    historical_observed_points: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="List of past observed coordinates"
    )


class DetectionResponse(BaseModel):
    detected: bool
    confidence: float
    latitude: float
    longitude: float
    bounding_box_pixels: Optional[Tuple[int, int, int, int]]


class IntensityResponse(BaseModel):
    category_code: str
    category_index: int
    confidence: float
    class_probabilities: Dict[str, float]
    maximum_sustained_wind_kts: float
    maximum_sustained_wind_kmph: float
    central_pressure_hpa: float
    radius_of_maximum_winds_km: float


class FullPipelineResponse(BaseModel):
    storm_id: str
    timestamp: str
    ocean_basin: Dict[str, Any]
    eye_detection: DetectionResponse
    intensity_classification: IntensityResponse
    spatiotemporal_forecast: Dict[str, Any]
    gis_layers: Dict[str, Any]
    synoptic_environment: Dict[str, float]


# ==========================================
# Endpoints
# ==========================================

@router.get("/regional-basins")
async def get_regional_basins():
    """Returns defined ocean basins and geographic bounding boxes."""
    bounds_path = "configs/regional_bounds.json"
    if os.path.exists(bounds_path):
        with open(bounds_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "error", "message": "Regional bounds config not found."}


@router.get("/live-storms")
async def get_live_storms():
    """
    Returns active or monitored cyclone vortex candidates across the North & South Indian Oceans.
    """
    sample_cyclones_path = "data/annotations/sample_cyclones.json"
    storms = []
    if os.path.exists(sample_cyclones_path):
        with open(sample_cyclones_path, "r", encoding="utf-8") as f:
            storms = json.load(f).get("historical_cyclones", [])

    # Add live environmental telemetry to each candidate
    for s in storms:
        track = s.get("track_sample", [])
        if track:
            last_pt = track[-1]
            live_env = _openmeteo_fetcher.fetch_live_marine_weather(last_pt["lat"], last_pt["lon"])
            s["latest_observed"] = last_pt
            s["live_marine_weather"] = live_env

    return {"active_storms": storms, "count": len(storms), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}


@router.post("/full-pipeline", response_model=FullPipelineResponse)
async def run_cyclone_prediction(
    req: InferenceRequest,
    background_tasks: BackgroundTasks,
    _rate_limit: bool = Depends(RateLimiter(requests_per_minute=60)),
    dispatcher: ModelDispatcher = Depends(get_dispatcher),
):
    """
    Executes the end-to-end Tropical Cyclone Deep Learning Pipeline:
    1. Basin Dispatching
    2. Synthetic or Live Multi-Channel Satellite Tensor Synthesis
    3. Eye Detection
    4. ResNet-50 + Dense Multimodal IMD Classification
    5. ConvLSTM Spatiotemporal Trajectory Forecasting
    6. Leaflet GeoJSON Vector Overlay Generation
    """
    lat = req.latitude
    lon = req.longitude

    # 1. Fetch / synthesize synoptic environmental vector
    syn_dict = req.synoptic_features.model_dump()
    if req.fetch_live_weather:
        live_w = _openmeteo_fetcher.fetch_live_marine_weather(lat, lon)
        syn_dict["sst_celsius"] = live_w.get("temperature_2m_celsius", syn_dict["sst_celsius"])
        syn_dict["mean_sea_level_pressure_hpa"] = live_w.get("surface_pressure_hpa", syn_dict["mean_sea_level_pressure_hpa"])
        syn_dict["relative_humidity_700_pct"] = live_w.get("relative_humidity_pct", syn_dict["relative_humidity_700_pct"])

    syn_tensor = _preprocessor.normalize_synoptic_vector(syn_dict).unsqueeze(0)

    # 2. Synthesize / fetch satellite raster centered on storm coordinates
    sat_raw = _mosdac_fetcher.fetch_or_synthesize_raster(
        datetime.datetime.now(datetime.timezone.utc),
        center_lat=lat,
        center_lon=lon,
        size=(256, 256),
        storm_intensity_knots=70.0,
    )
    sat_tensor = _preprocessor.normalize_satellite_tensor(sat_raw).unsqueeze(0)

    # 3. Run Pipeline through Model Dispatcher
    result = dispatcher.run_full_pipeline(
        satellite_tensor=sat_tensor,
        synoptic_vector=syn_tensor,
        approx_lat=lat,
        approx_lon=lon,
    )

    # 4. Generate GIS Vector Layers for Interactive Leaflet / Mapbox
    observed_pts = req.historical_observed_points or [
        {"lat": lat - 1.2, "lon": lon - 1.5, "msw_kts": 50, "timestamp": "-12h"},
        {"lat": lat - 0.6, "lon": lon - 0.8, "msw_kts": 65, "timestamp": "-6h"},
        {"lat": lat, "lon": lon, "msw_kts": 80, "timestamp": "0h (Now)"},
    ]

    forecast_pts = result["spatiotemporal_forecast"]["track"]
    gis_payload = create_leaflet_vector_payload(
        observed_track=observed_pts,
        forecast_track=forecast_pts,
        center_lat=result["eye_detection"]["latitude"],
        center_lon=result["eye_detection"]["longitude"],
        max_wind_kts=result["intensity_classification"]["maximum_sustained_wind_kts"],
    )

    storm_id = f"CYC_{int(datetime.datetime.now().timestamp())}_{result['basin']['id']}"

    response_payload = {
        "storm_id": storm_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ocean_basin": result["basin"],
        "eye_detection": result["eye_detection"],
        "intensity_classification": result["intensity_classification"],
        "spatiotemporal_forecast": result["spatiotemporal_forecast"],
        "gis_layers": gis_payload,
        "synoptic_environment": syn_dict,
    }

    # Fire-and-forget: alert any WhatsApp subscribers within range of this
    # storm. Runs after the response is sent, and failures here never affect
    # the prediction response itself.
    background_tasks.add_task(broadcast_alert_for_prediction, response_payload)

    return response_payload