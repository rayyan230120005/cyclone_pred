import React from 'react';
import { TrendingUp, BarChart3, AlertCircle } from 'lucide-react';

const IMD_PALETTE = {
  D: '#38bdf8',
  DD: '#0284c7',
  CS: '#eab308',
  SCS: '#f97316',
  VSCS: '#ef4444',
  ESCS: '#a855f7',
  SuCS: '#ec4899',
};

export default function IntensityChart({ intensityData, forecastData }) {
  const probs = intensityData?.class_probabilities || {
    D: 0.01,
    DD: 0.02,
    CS: 0.05,
    SCS: 0.12,
    VSCS: 0.72,
    ESCS: 0.07,
    SuCS: 0.01,
  };

  const track = forecastData?.track || [];

  return (
    <div className="glass-panel p-4 flex flex-col gap-4">
      {/* Probability Distribution */}
      <div>
        <div className="flex items-center justify-between text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
          <span className="flex items-center gap-1.5"><BarChart3 className="w-4 h-4 text-cyan-400" /> IMD Category Probabilities</span>
          <span className="text-slate-400">Softmax Confidence</span>
        </div>
        
        <div className="grid grid-cols-7 gap-1.5">
          {Object.entries(probs).map(([cat, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const isHighest = cat === intensityData?.category_code;
            return (
              <div key={cat} className="flex flex-col items-center gap-1">
                <div className="text-[11px] font-bold" style={{ color: IMD_PALETTE[cat] || '#38bdf8' }}>
                  {cat}
                </div>
                <div className="w-full bg-slate-800/80 rounded h-20 flex items-end p-0.5 overflow-hidden">
                  <div
                    className="w-full rounded transition-all duration-500"
                    style={{
                      height: `${Math.max(4, prob * 100)}%`,
                      backgroundColor: IMD_PALETTE[cat] || '#38bdf8',
                      boxShadow: isHighest ? `0 0 10px ${IMD_PALETTE[cat]}` : 'none',
                    }}
                  ></div>
                </div>
                <div className="text-[10px] text-slate-400 font-mono">{pct}%</div>
              </div>
            );
          })}
        </div>
      </div>

      <hr className="border-slate-800" />

      {/* Spatiotemporal Track Forecast Timeline */}
      <div>
        <div className="flex items-center justify-between text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2.5">
          <span className="flex items-center gap-1.5"><TrendingUp className="w-4 h-4 text-rose-400" /> 48-Hour Spatiotemporal Trajectory</span>
          <span className="text-slate-400 text-[11px]">ConvLSTM Sequence Model</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {track.slice(0, 4).map((step, idx) => (
            <div key={idx} className="bg-slate-900/80 border border-slate-800 p-2.5 rounded-lg flex flex-col gap-1">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-bold text-rose-400">{step.forecast_time_offset}</span>
                <span className="text-slate-400 font-mono">±{step.uncertainty_radius_km}km</span>
              </div>
              <div className="text-sm font-extrabold text-white hud-font">
                {step.estimated_wind_kts} <span className="text-[10px] font-normal text-slate-400">kts</span>
              </div>
              <div className="text-[11px] text-slate-400 font-mono truncate">
                {step.latitude}°N, {step.longitude}°E
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
