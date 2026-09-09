"""
Indian Meteorological Department (IMD) Cyclone Metrics & Trajectory Error Evaluators.

Includes:
- Haversine great-circle distance
- Along-Track Error (ATE) & Cross-Track Error (CTE)
- Direct Position Error (DPE)
- IMD intensity scale mapping and wind-pressure relationships
- Brier Skill Score for probabilistic forecasting
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


@dataclass
class IMDIntensityScale:
    code: str
    name: str
    min_kts: float
    max_kts: float
    min_kmph: float
    max_kmph: float
    approx_central_pressure_drop_hpa: str


IMD_SCALE_TABLE: List[IMDIntensityScale] = [
    IMDIntensityScale("D", "Depression", 17, 27, 31, 49, "1.5 - 3.0"),
    IMDIntensityScale("DD", "Deep Depression", 28, 33, 50, 61, "3.0 - 4.5"),
    IMDIntensityScale("CS", "Cyclonic Storm", 34, 47, 62, 88, "4.5 - 8.5"),
    IMDIntensityScale("SCS", "Severe Cyclonic Storm", 48, 63, 89, 117, "8.5 - 15.5"),
    IMDIntensityScale("VSCS", "Very Severe Cyclonic Storm", 64, 89, 118, 166, "15.5 - 31.5"),
    IMDIntensityScale("ESCS", "Extremely Severe Cyclonic Storm", 90, 119, 167, 221, "31.5 - 65.5"),
    IMDIntensityScale("SuCS", "Super Cyclonic Storm", 120, 250, 222, 450, ">= 65.6"),
]


def classify_imd_category(wind_speed_kts: float) -> IMDIntensityScale:
    """
    Returns the IMD category object corresponding to given wind speed in knots.
    """
    for cat in reversed(IMD_SCALE_TABLE):
        if wind_speed_kts >= cat.min_kts:
            return cat
    return IMD_SCALE_TABLE[0]


def calculate_haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two coordinates using the Haversine formula (Earth radius = 6371 km).
    """
    r_earth = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def calculate_along_and_cross_track_error(
    obs_start: Tuple[float, float],
    obs_end: Tuple[float, float],
    forecast_point: Tuple[float, float],
) -> Dict[str, float]:
    """
    Decomposes position error into Along-Track Error (ATE) and Cross-Track Error (CTE).
    - ATE: Error parallel to storm track direction (+ indicates storm forecast is ahead of actual, - indicates behind)
    - CTE: Error perpendicular to track (+ indicates right of track, - indicates left)
    - DPE: Direct Position Error (Euclidean Haversine distance)
    """
    lat1, lon1 = obs_start
    lat2, lon2 = obs_end
    f_lat, f_lon = forecast_point

    # Direct Position Error to target end point
    dpe = calculate_haversine_distance_km(lat2, lon2, f_lat, f_lon)

    # Segment length
    seg_dist = calculate_haversine_distance_km(lat1, lon1, lat2, lon2)
    if seg_dist < 1e-3:
        return {"direct_position_error_km": round(dpe, 2), "along_track_error_km": 0.0, "cross_track_error_km": round(dpe, 2)}

    # Distance from start to forecast
    d_start_f = calculate_haversine_distance_km(lat1, lon1, f_lat, f_lon)

    # Initial bearing from start to end (radians)
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    bearing_track = math.atan2(y, x)

    # Bearing from start to forecast point
    y_f = math.sin(math.radians(f_lon - lon1)) * math.cos(math.radians(f_lat))
    x_f = math.cos(math.radians(lat1)) * math.sin(math.radians(f_lat)) - math.sin(math.radians(lat1)) * math.cos(math.radians(f_lat)) * math.cos(math.radians(f_lon - lon1))
    bearing_f = math.atan2(y_f, x_f)

    angle_diff = bearing_f - bearing_track

    # Cross-track error (perpendicular)
    cte = math.asin(math.sin(d_start_f / 6371.0) * math.sin(angle_diff)) * 6371.0

    # Along-track distance from start
    along_dist = math.acos(math.cos(d_start_f / 6371.0) / max(1e-6, math.cos(cte / 6371.0))) * 6371.0
    ate = along_dist - seg_dist

    return {
        "direct_position_error_km": round(dpe, 2),
        "along_track_error_km": round(ate, 2),
        "cross_track_error_km": round(cte, 2),
    }


def calculate_brier_score(forecast_probs: np.ndarray, ground_truth_one_hot: np.ndarray) -> float:
    """
    Computes multi-class Brier Score: BS = (1/N) * sum_i sum_k (p_ik - o_ik)^2
    Lower is better (0.0 is perfect).
    """
    return float(np.mean(np.sum((forecast_probs - ground_truth_one_hot) ** 2, axis=-1)))


if __name__ == "__main__":
    dist = calculate_haversine_distance_km(12.0, 85.0, 15.0, 88.0)
    print(f"Haversine distance (12,85) to (15,88): {dist:.2f} km")
    cat = classify_imd_category(115.0)
    print(f"115 kts classification: {cat.code} - {cat.name}")
    decomp = calculate_along_and_cross_track_error((10.0, 85.0), (12.0, 86.0), (12.2, 85.8))
    print(f"Track Error Decomposition: {decomp}")
