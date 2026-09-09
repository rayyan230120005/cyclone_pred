import React from 'react';
import { Radio, Activity, Sparkles, Navigation } from 'lucide-react';

export default function LiveFeed({ activeStorms, onSelectStorm, selectedStormId, isWsConnected }) {
  return (
    <div className="glass-panel p-4 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Active Ocean Telemetry & Storms
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px]">
          <span className={`w-2 h-2 rounded-full ${isWsConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
          <span className="text-slate-400">{isWsConnected ? 'Live Stream Active' : 'Polling (HTTP)'}</span>
        </div>
      </div>

      {/* Storm List */}
      <div className="flex flex-col gap-2 max-h-[300px] overflow-y-auto pr-1">
        {activeStorms?.length > 0 ? (
          activeStorms.map((storm) => {
            const isSelected = storm.id === selectedStormId;
            const lastTrack = storm.latest_observed || storm.track_sample?.[storm.track_sample.length - 1];

            return (
              <button
                key={storm.id}
                onClick={() => onSelectStorm(storm)}
                className={`w-full text-left p-2.5 rounded-lg border transition-all duration-200 flex flex-col gap-1.5 ${
                  isSelected
                    ? 'bg-cyan-950/40 border-cyan-400/80 shadow-[0_0_12px_rgba(56,189,248,0.25)]'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-850'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-sm text-white flex items-center gap-1.5 truncate">
                    <Radio className={`w-3.5 h-3.5 ${isSelected ? 'text-cyan-400' : 'text-slate-500'}`} />
                    {storm.name}
                  </span>
                  <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-rose-500/20 text-rose-300 border border-rose-500/40">
                    {storm.peak_intensity_category || 'VSCS'}
                  </span>
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400">
                  <span>Basin: <b className="text-slate-300">{storm.basin_code || 'BOB'}</b></span>
                  <span>Peak: <b className="text-cyan-300">{storm.peak_wind_kts || 115} kts</b></span>
                </div>

                {lastTrack && (
                  <div className="flex items-center gap-1 text-[10px] text-slate-500">
                    <Navigation className="w-3 h-3 text-slate-500" />
                    <span>Eye at {lastTrack.lat}°N, {lastTrack.lon}°E</span>
                  </div>
                )}
              </button>
            );
          })
        ) : (
          <div className="text-xs text-slate-500 py-4 text-center">
            No active storms detected in current satellite pass.
          </div>
        )}
      </div>
    </div>
  );
}
