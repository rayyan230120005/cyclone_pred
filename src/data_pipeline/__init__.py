"""
Data ingestion, satellite fetching, climate reanalysis, and spatial preprocessing pipeline.
"""
from .fetch_mosdac_insat import MosdacInsatFetcher
from .fetch_copernicus_era5 import CopernicusEra5Fetcher
from .fetch_nasa_earthdata import NasaEarthdataFetcher
from .fetch_openmeteo import OpenMeteoFetcher
from .spatial_aligner import SpatialAligner
from .preprocessor import MultimodalPreprocessor

__all__ = [
    "MosdacInsatFetcher",
    "CopernicusEra5Fetcher",
    "NasaEarthdataFetcher",
    "OpenMeteoFetcher",
    "SpatialAligner",
    "MultimodalPreprocessor",
]
