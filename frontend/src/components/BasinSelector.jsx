import React from 'react';

export default function BasinSelector({
  activeBasin,
  onSelectBasin,
  lat,
  lon,
  onLatChange,
  onLonChange,
  sst,
  onSstChange,
  shear,
  onShearChange,
  onTriggerInference,
  isLoading,
}) {
  const basins = [
    { id: 'bay_of_bengal', name: 'Bay of Bengal', code: 'BOB', defaultLat: 15.2, defaultLon: 88.4 },
    { id: 'arabian_sea', name: 'Arabian Sea', code: 'ARB', defaultLat: 16.5, defaultLon: 68.2 },
    { id: 'indian_ocean_south', name: 'South Indian Ocean', code: 'SIO', defaultLat: -16.0, defaultLon: 75.0 },
  ];

  return (
    <div className="glass-panel p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-sm">◉</span>
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Ocean Basin Routing & Parameters
          </span>
        </div>
        <span className="text-[11px] text-cyan-400 font-mono">Auto Dispatcher</span>
      </div>

      {/* Basin Selector Tabs */}
      <div className="grid grid-cols-3 gap-2">
        {basins.map((b) => {
          const isActive = activeBasin === b.id;
          return (
            <button
              key={b.id}
              onClick={() => {
                onSelectBasin(b.id);
                onLatChange(b.defaultLat);
                onLonChange(b.defaultLon);
              }}
              className={`py-2 px-1 rounded-lg text-xs font-semibold transition-all duration-200 flex flex-col items-center gap-0.5 border ${
                isActive
                  ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-[0_0_10px_rgba(56,189,248,0.3)]'
                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              <span>{b.code}</span>
              <span className="text-[10px] font-normal text-slate-400 truncate max-w-full">{b.name}</span>
            </button>
          );
        })}
      </div>

      {/* Coordinate Inputs */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <label className="text-slate-400 block mb-1">Latitude (°)</label>
          <input
            type="number"
            step="0.1"
            value={lat}
            onChange={(e) => onLatChange(parseFloat(e.target.value) || 0)}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:outline-none focus:border-cyan-400"
          />
        </div>
        <div>
          <label className="text-slate-400 block mb-1">Longitude (°)</label>
          <input
            type="number"
            step="0.1"
            value={lon}
            onChange={(e) => onLonChange(parseFloat(e.target.value) || 0)}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:outline-none focus:border-cyan-400"
          />
        </div>
      </div>

      {/* Atmospheric Slider Overrides */}
      <div className="flex flex-col gap-2.5 text-xs">
        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>Sea Surface Temp (SST)</span>
            <span className="text-cyan-300 font-mono">{sst}°C</span>
          </div>
          <input
            type="range"
            min="25.0"
            max="32.5"
            step="0.1"
            value={sst}
            onChange={(e) => onSstChange(parseFloat(e.target.value))}
            className="w-full accent-cyan-400"
          />
        </div>

        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>850-200 hPa Wind Shear</span>
            <span className="text-cyan-300 font-mono">{shear} kts</span>
          </div>
          <input
            type="range"
            min="4.0"
            max="45.0"
            step="0.5"
            value={shear}
            onChange={(e) => onShearChange(parseFloat(e.target.value))}
            className="w-full accent-cyan-400"
          />
        </div>
      </div>

      {/* Run Inference Action Button */}
      <button
        onClick={onTriggerInference}
        disabled={isLoading}
        className="btn-primary w-full py-2.5 text-sm disabled:opacity-50"
      >
        <span>{isLoading ? 'Synthesizing & Inferencing...' : 'Execute Multimodal Forecast'}</span>
      </button>
    </div>
  );
}
