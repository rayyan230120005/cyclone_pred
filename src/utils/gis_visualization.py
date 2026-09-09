"""
GIS Vector Layer & GeoJSON Visualizer for Mapbox / Leaflet Dashboards.

Generates:
- GeoJSON LineString and Point FeatureCollections for observed and forecast storm tracks
- Expanding Cone of Uncertainty polygon envelope
- Quadrant-based Wind Radii buffers (R34, R50, R64 gale/storm/hurricane force winds)
- Basin Boundary polygons for interactive map overlays
"""

import math
from typing import List, Dict, Any, Tuple


# IMD Color Palette
IMD_COLORS = {
    "D": "#38bdf8",     # Light Blue (Depression)
    "DD": "#0284c7",    # Deep Blue (Deep Depression)
    "CS": "#eab308",    # Yellow (Cyclonic Storm)
    "SCS": "#f97316",   # Orange (Severe Cyclonic Storm)
    "VSCS": "#ef4444",  # Red (Very Severe Cyclonic Storm)
    "ESCS": "#a855f7",  # Purple (Extremely Severe Cyclonic Storm)
    "SuCS": "#ec4899",  # Magenta/Pink (Super Cyclonic Storm)
}


def generate_geojson_track(track_points: List[Dict[str, Any]], track_type: str = "forecast") -> Dict[str, Any]:
    """
    Converts a list of track points into a GeoJSON FeatureCollection with points and LineString.
    """
    features = []
    coordinates = []

    for idx, pt in enumerate(track_points):
        lat = pt.get("latitude", pt.get("lat"))
        lon = pt.get("longitude", pt.get("lon"))
        coordinates.append([lon, lat])

        wind_kts = pt.get("estimated_wind_kts", pt.get("msw_kts", 35.0))
        wind_kmph = pt.get("estimated_wind_kmph", round(wind_kts * 1.852, 1))
        lead_time = pt.get("forecast_time_offset", pt.get("timestamp", f"+{idx * 6}h"))

        # Determine category code
        if wind_kts >= 120:
            cat = "SuCS"
        elif wind_kts >= 90:
            cat = "ESCS"
        elif wind_kts >= 64:
            cat = "VSCS"
        elif wind_kts >= 48:
            cat = "SCS"
        elif wind_kts >= 34:
            cat = "CS"
        elif wind_kts >= 28:
            cat = "DD"
        else:
            cat = "D"

        point_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "step": idx,
                "lead_time": lead_time,
                "wind_kts": wind_kts,
                "wind_kmph": wind_kmph,
                "category": cat,
                "color": IMD_COLORS.get(cat, "#38bdf8"),
                "uncertainty_km": pt.get("uncertainty_radius_km", 20.0),
                "track_type": track_type,
            },
        }
        features.append(point_feature)

    # LineString connecting the track points
    if len(coordinates) > 1:
        line_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "track_type": track_type,
                "stroke_color": "#f43f5e" if track_type == "forecast" else "#38bdf8",
                "stroke_width": 3.5,
                "stroke_dash": [6, 6] if track_type == "forecast" else None,
            },
        }
        features.insert(0, line_feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def generate_uncertainty_cone_geojson(track_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates an expanding polygon envelope (Cone of Uncertainty) around forecast trajectory.
    """
    if len(track_points) < 2:
        return {"type": "FeatureCollection", "features": []}

    left_boundary = []
    right_boundary = []

    for i in range(len(track_points)):
        pt = track_points[i]
        lat = pt.get("latitude", pt.get("lat"))
        lon = pt.get("longitude", pt.get("lon"))
        r_km = pt.get("uncertainty_radius_km", max(25.0, 15.0 + 3.0 * (i * 6)))

        # Convert km to rough degrees (1 deg lat ~ 111 km)
        r_deg_lat = r_km / 111.0
        r_deg_lon = r_km / (111.0 * max(0.1, math.cos(math.radians(lat))))

        # Determine direction vector for perpendicular expansion
        if i < len(track_points) - 1:
            next_pt = track_points[i + 1]
            n_lat = next_pt.get("latitude", next_pt.get("lat"))
            n_lon = next_pt.get("longitude", next_pt.get("lon"))
            dx = n_lon - lon
            dy = n_lat - lat
        else:
            prev_pt = track_points[i - 1]
            p_lat = prev_pt.get("latitude", prev_pt.get("lat"))
            p_lon = prev_pt.get("longitude", prev_pt.get("lon"))
            dx = lon - p_lon
            dy = lat - p_lat

        mag = math.sqrt(dx * dx + dy * dy) or 1e-5
        # Perpendicular unit vector
        perp_x = -dy / mag
        perp_y = dx / mag

        left_lon = lon + perp_x * r_deg_lon
        left_lat = lat + perp_y * r_deg_lat
        right_lon = lon - perp_x * r_deg_lon
        right_lat = lat - perp_y * r_deg_lat

        left_boundary.append([left_lon, left_lat])
        right_boundary.append([right_lon, right_lat])

    # Construct enclosed polygon coordinates
    polygon_coords = left_boundary + list(reversed(right_boundary)) + [left_boundary[0]]

    cone_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon_coords],
        },
        "properties": {
            "layer_type": "uncertainty_cone",
            "fill": "#fb7185",
            "fill_opacity": 0.22,
            "stroke": "#f43f5e",
            "stroke_width": 1.5,
            "stroke_opacity": 0.6,
        },
    }

    return {
        "type": "FeatureCollection",
        "features": [cone_feature],
    }


def generate_wind_radii_geojson(center_lat: float, center_lon: float, max_wind_kts: float) -> Dict[str, Any]:
    """
    Generates circular / quadrant wind radii buffers for R34, R50, and R64 wind thresholds.
    """
    radii_specs = [
        {"name": "R34 (Gale Force >= 34 kts)", "threshold": 34, "radius_km": min(280.0, max_wind_kts * 2.2), "color": "#facc15", "opacity": 0.15},
        {"name": "R50 (Storm Force >= 50 kts)", "threshold": 50, "radius_km": min(160.0, max_wind_kts * 1.3), "color": "#fb923c", "opacity": 0.20},
        {"name": "R64 (Hurricane Force >= 64 kts)", "threshold": 64, "radius_km": min(80.0, max_wind_kts * 0.7), "color": "#f43f5e", "opacity": 0.25},
    ]

    features = []
    for spec in radii_specs:
        if max_wind_kts >= spec["threshold"]:
            r_km = spec["radius_km"]
            # Generate circle polygon with 36 vertices
            poly_points = []
            for deg in range(0, 361, 10):
                rad = math.radians(deg)
                d_lat = (r_km / 111.0) * math.sin(rad)
                d_lon = (r_km / (111.0 * max(0.1, math.cos(math.radians(center_lat))))) * math.cos(rad)
                poly_points.append([center_lon + d_lon, center_lat + d_lat])

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly_points],
                },
                "properties": {
                    "layer_type": "wind_radii",
                    "wind_threshold_kts": spec["threshold"],
                    "label": spec["name"],
                    "radius_km": round(r_km, 1),
                    "fill": spec["color"],
                    "fill_opacity": spec["opacity"],
                    "stroke": spec["color"],
                    "stroke_width": 1.2,
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def create_leaflet_vector_payload(
    observed_track: List[Dict[str, Any]],
    forecast_track: List[Dict[str, Any]],
    center_lat: float,
    center_lon: float,
    max_wind_kts: float,
) -> Dict[str, Any]:
    """
    Packages all GIS vector layers into a unified JSON structure for direct Leaflet map consumption.
    """
    return {
        "observed_track_geojson": generate_geojson_track(observed_track, track_type="observed"),
        "forecast_track_geojson": generate_geojson_track(forecast_track, track_type="forecast"),
        "uncertainty_cone_geojson": generate_uncertainty_cone_geojson(forecast_track),
        "wind_radii_geojson": generate_wind_radii_geojson(center_lat, center_lon, max_wind_kts),
        "map_center": [center_lat, center_lon],
        "default_zoom": 6,
    }


if __name__ == "__main__":
    dummy_forecast = [
        {"latitude": 15.0, "longitude": 88.0, "estimated_wind_kts": 65.0, "uncertainty_radius_km": 25.0},
        {"latitude": 16.2, "longitude": 87.8, "estimated_wind_kts": 80.0, "uncertainty_radius_km": 45.0},
        {"latitude": 17.8, "longitude": 87.5, "estimated_wind_kts": 95.0, "uncertainty_radius_km": 75.0},
        {"latitude": 19.5, "longitude": 87.1, "estimated_wind_kts": 105.0, "uncertainty_radius_km": 110.0},
    ]
    cone = generate_uncertainty_cone_geojson(dummy_forecast)
    print(f"Generated uncertainty cone feature count: {len(cone['features'])}")
    radii = generate_wind_radii_geojson(15.0, 88.0, 85.0)
    print(f"Generated wind radii rings: {len(radii['features'])}")
