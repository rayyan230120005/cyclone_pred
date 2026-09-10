import React from 'react';

const CATEGORY_COLORS = {
  D: { bg: 'bg-sky-400', text: 'text-black', name: 'Depression', border: 'border-sky-400' },
  DD: { bg: 'bg-sky-600', text: 'text-white', name: 'Deep Depression', border: 'border-sky-500' },
  CS: { bg: 'bg-amber-400', text: 'text-black', name: 'Cyclonic Storm', border: 'border-amber-400' },
  SCS: { bg: 'bg-orange-500', text: 'text-white', name: 'Severe Cyclonic Storm', border: 'border-orange-500' },
  VSCS: { bg: 'bg-rose-500', text: 'text-white', name: 'Very Severe Cyclonic Storm', border: 'border-rose-500' },
  ESCS: { bg: 'bg-purple-500', text: 'text-white', name: 'Extremely Severe Cyclonic Storm', border: 'border-purple-500' },
  SuCS: { bg: 'bg-pink-500', text: 'text-white', name: 'Super Cyclonic Storm', border: 'border-pink-500' },
};

export default function Gauges({ intensityData, eyeData, synopticData }) {
  const catCode = intensityData?.category_code || 'VSCS';
  const catStyle = CATEGORY_COLORS[catCode] || CATEGORY_COLORS['VSCS'];
  const windKts = intensityData?.maximum_sustained_wind_kts || 80.0;
  const windKmph = intensityData?.maximum_sustained_wind_kmph || Math.round(windKts * 1.852);
  const pressure = intensityData?.central_pressure_hpa || 960.0;
  const rmw = intensityData?.radius_of_maximum_winds_km || 35.0;
  const confidence = (intensityData?.confidence * 100 || 94.2).toFixed(1);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
      {/* 1. IMD Classification Card */}
      <div className={`glass-panel p-4 flex flex-col justify-between border-l-4 ${catStyle.border}`}>
        <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-semibold tracking-wider">
          <span>IMD Classification</span>
          <span className="text-cyan-400">{confidence}% Conf</span>
        </div>
        <div className="my-2">
          <div className="flex items-baseline gap-2">
            <span className={`px-2.5 py-0.5 rounded text-lg font-bold hud-font ${catStyle.bg} ${catStyle.text}`}>
              {catCode}
            </span>
            <span className="text-sm font-medium text-slate-200">{catStyle.name}</span>
          </div>
        </div>
        <div className="text-[11px] text-slate-400">
          Multimodal ResNet-50 + Dense Fusion Inference
        </div>
      </div>

      {/* 2. Maximum Sustained Wind Card */}
      <div className="glass-panel p-4 flex flex-col justify-between border-l-4 border-cyan-400">
        <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-semibold tracking-wider">
          <span>Max Sustained Wind</span>
          <span className="text-cyan-300 hud-font">{windKmph} km/h</span>
        </div>
        <div className="my-1">
          <div className="text-3xl font-extrabold text-white hud-font flex items-baseline gap-1">
            {windKts} <span className="text-sm font-normal text-slate-400 font-sans">knots (3-min avg)</span>
          </div>
        </div>
        {/* Progress bar towards Super Cyclone threshold (120 kts) */}
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1">
          <div
            className="bg-gradient-to-r from-cyan-400 via-amber-400 to-rose-500 h-full rounded-full"
            style={{ width: `${Math.min(100, (windKts / 130) * 100)}%` }}
          ></div>
        </div>
      </div>

      {/* 3. Central Pressure Card */}
      <div className="glass-panel p-4 flex flex-col justify-between border-l-4 border-purple-400">
        <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-semibold tracking-wider">
          <span>Central Pressure</span>
          <span className="text-purple-300">MSLP</span>
        </div>
        <div className="my-1">
          <div className="text-3xl font-extrabold text-white hud-font flex items-baseline gap-1">
            {pressure} <span className="text-sm font-normal text-slate-400 font-sans">hPa / mbar</span>
          </div>
        </div>
        <div className="text-[11px] text-slate-400 flex justify-between">
          <span>Drop: -{(1010 - pressure).toFixed(1)} hPa</span>
          <span>RMW: {rmw} km</span>
        </div>
      </div>

      {/* 4. Atmospheric SST & Shear Card */}
      <div className="glass-panel p-4 flex flex-col justify-between border-l-4 border-emerald-400">
        <div className="flex items-center justify-between text-slate-400 text-xs uppercase font-semibold tracking-wider">
          <span>Marine Environment</span>
          <span className="text-emerald-300">ERA5 Reanalysis</span>
        </div>
        <div className="grid grid-cols-2 gap-2 my-1">
          <div>
            <div className="text-xs text-slate-400">SST</div>
            <div className="text-lg font-bold text-white hud-font">
              {synopticData?.sst_celsius || 29.5}°C
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Wind Shear</div>
            <div className="text-lg font-bold text-white hud-font">
              {synopticData?.vertical_wind_shear_kts || 12.0} kts
            </div>
          </div>
        </div>
        <div className="text-[11px] text-emerald-400/90 font-medium">
          Thermodynamically Favorable for Intensification
        </div>
      </div>
    </div>
  );
}
