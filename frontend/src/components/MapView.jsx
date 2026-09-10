import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Polygon, Circle, useMap } from 'react-leaflet';

// Helper component to center map dynamically
function MapCenterController({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center && center.length === 2) {
      map.setView(center, map.getZoom());
    }
  }, [center, map]);
  return null;
}

export default function MapView({ predictionData, center, activeLayers }) {
  const mapCenter = center || [15.0, 88.0];
  const gis = predictionData?.gis_layers;
  const forecastTrack = predictionData?.spatiotemporal_forecast?.track || [];
  const eye = predictionData?.eye_detection;
  const intensity = predictionData?.intensity_classification;

  // Extract Cone Coordinates
  const coneFeature = gis?.uncertainty_cone_geojson?.features?.[0];
  const coneCoords = coneFeature?.geometry?.coordinates?.[0]?.map(c => [c[1], c[0]]) || [];

  // Extract Track Coordinates
  const trackLine = forecastTrack.map(pt => [pt.latitude, pt.longitude]);
  if (eye?.latitude && eye?.longitude) {
    trackLine.unshift([eye.latitude, eye.longitude]);
  }

  return (
    <div className="relative w-full h-[520px] rounded-xl overflow-hidden glass-panel border border-cyan-500/20">
      <MapContainer
        center={mapCenter}
        zoom={6}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <MapCenterController center={mapCenter} />

        {/* OpenStreetMap base layer with no API key required */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Optional NASA GIBS Real-Time Satellite Layer */}
        {activeLayers?.satellite && (
          <TileLayer
            attribution="NASA EOSDIS GIBS"
            url="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
            opacity={0.65}
          />
        )}

        {/* Uncertainty Cone Polygon */}
        {activeLayers?.uncertaintyCone && coneCoords.length > 0 && (
          <Polygon
            positions={coneCoords}
            pathOptions={{
              color: '#f43f5e',
              fillColor: '#fb7185',
              fillOpacity: 0.2,
              weight: 1.5,
              dashArray: '4, 4',
            }}
          />
        )}

        {/* Wind Radii Buffers */}
        {activeLayers?.windRadii && eye && intensity && (
          <>
            <Circle
              center={[eye.latitude, eye.longitude]}
              radius={(intensity.radius_of_maximum_winds_km || 30) * 1000}
              pathOptions={{ color: '#f43f5e', fillColor: '#f43f5e', fillOpacity: 0.25, weight: 1.5 }}
            />
            <Circle
              center={[eye.latitude, eye.longitude]}
              radius={180000}
              pathOptions={{ color: '#facc15', fillColor: '#facc15', fillOpacity: 0.1, weight: 1 }}
            />
          </>
        )}

        {/* Forecast Track Line */}
        {trackLine.length > 1 && (
          <Polyline
            positions={trackLine}
            pathOptions={{
              color: '#38bdf8',
              weight: 3.5,
              dashArray: '6, 6',
              opacity: 0.85,
            }}
          />
        )}

        {/* Forecast Step Markers */}
        {forecastTrack.map((pt, idx) => (
          <Circle
            key={idx}
            center={[pt.latitude, pt.longitude]}
            radius={pt.uncertainty_radius_km * 1000}
            pathOptions={{
              color: '#fb7185',
              fillColor: '#fb7185',
              fillOpacity: 0.12,
              weight: 1,
            }}
          >
            <Popup>
              <div className="text-slate-900 font-sans p-1">
                <div className="font-bold text-sm text-rose-600">Forecast {pt.forecast_time_offset}</div>
                <div><b>Position:</b> {pt.latitude}°N, {pt.longitude}°E</div>
                <div><b>Wind Speed:</b> {pt.estimated_wind_kts} kts ({pt.estimated_wind_kmph} km/h)</div>
                <div><b>Uncertainty Radius:</b> ±{pt.uncertainty_radius_km} km</div>
              </div>
            </Popup>
          </Circle>
        ))}

        {/* Active Cyclone Eye Marker */}
        {eye && (
          <Marker position={[eye.latitude, eye.longitude]}>
            <Popup>
              <div className="text-slate-900 font-sans p-1">
                <div className="font-bold text-sm text-cyan-600">Active Cyclone Center</div>
                <div><b>Intensity:</b> {intensity?.category_code}</div>
                <div><b>Max Wind:</b> {intensity?.maximum_sustained_wind_kts} kts</div>
                <div><b>Central Pressure:</b> {intensity?.central_pressure_hpa} hPa</div>
                <div><b>Confidence:</b> {(intensity?.confidence * 100 || 92).toFixed(1)}%</div>
              </div>
            </Popup>
          </Marker>
        )}
      </MapContainer>

      {/* Map HUD Legend */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-slate-900/90 backdrop-blur-md p-2.5 rounded-lg border border-slate-700 text-xs flex flex-col gap-1.5 shadow-lg">
        <div className="font-semibold text-slate-300 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400"></span> GIS Overlay Legend
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-0.5 bg-cyan-400 border-dashed"></span>
          <span className="text-slate-400">48h ConvLSTM Trajectory</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-2 bg-rose-500/30 border border-rose-500 rounded-sm"></span>
          <span className="text-slate-400">Track Uncertainty Cone</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
          <span className="text-slate-400">Vortex Eye Center (YOLOv8)</span>
        </div>
      </div>
    </div>
  );
}
