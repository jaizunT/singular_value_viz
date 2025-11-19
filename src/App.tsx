import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Play, RefreshCw, Settings, Plus, Trash2, Info, Maximize, PenTool, Activity, X, Scaling, ChevronRight } from 'lucide-react';

/**
 * ------------------------------------------------------------------
 * POLAR EXPRESS CORE LOGIC
 * ------------------------------------------------------------------
 */

type CoeffTuple = [number, number, number];

const DEFAULT_RAW_COEFFS: CoeffTuple[] = [
  [8.28721201814563, -23.595886519098837, 17.300387312530933],
  [4.107059111542203, -2.9478499167379106, 0.5448431082926601],
  [3.9486908534822946, -2.908902115962949, 0.5518191394370137],
  [3.3184196573706015, -2.488488024314874, 0.51004894012372],
  [2.300652019954817, -1.6689039845747493, 0.4188073119525673],
  [1.891301407787398, -1.2679958271945868, 0.37680408948524835],
  [1.8750014808534479, -1.2500016453999487, 0.3750001645474248],
  [1.875, -1.25, 0.375], // limiting form
];

const DEFAULT_SAFETY = 1.01;
const DEFAULT_CUSHION = 0.024;

function getCoeffsForConfig(
  numIters: number,
  safety: number = DEFAULT_SAFETY,
  cushion: number = DEFAULT_CUSHION
): CoeffTuple[] {
  let scaled: CoeffTuple[] = DEFAULT_RAW_COEFFS.map((triple, i) => {
    const [a, b, c] = triple;
    if (i === DEFAULT_RAW_COEFFS.length - 1) {
      return [a, b, c];
    } else {
      const a_s = a / safety;
      const b_s = b / Math.pow(safety, 3);
      const c_s = c / Math.pow(safety, 5);
      return [a_s, b_s, c_s];
    }
  });

  if (cushion > 0) {
    const scaleC = cushion / DEFAULT_CUSHION;
    scaled = scaled.map(([a, b, c]) => [a, b, c * scaleC]);
  }

  if (numIters <= scaled.length) {
    return scaled.slice(0, numIters);
  } else {
    const last = scaled[scaled.length - 1];
    const extra = Array(numIters - scaled.length).fill(last);
    return [...scaled, ...extra];
  }
}

function updateSigmas(sigmas: Float64Array, coeffs: CoeffTuple[]): Float64Array[] {
  const history: Float64Array[] = [new Float64Array(sigmas)];
  let currentSigmas = new Float64Array(sigmas);

  for (const [a, b, c] of coeffs) {
    const nextSigmas = new Float64Array(currentSigmas.length);
    for (let i = 0; i < currentSigmas.length; i++) {
      const s = currentSigmas[i];
      const lam = s * s;
      const scale = a + b * lam + c * (lam * lam);
      nextSigmas[i] = scale * s;
    }
    currentSigmas = nextSigmas;
    history.push(currentSigmas);
  }
  return history;
}

/**
 * ------------------------------------------------------------------
 * DATA GENERATION & HELPERS
 * ------------------------------------------------------------------
 */

interface SpectrumPeak {
  id: string;
  mean: number;
  std: number;
  weight: number;
}

interface SketchPoint {
  x: number;
  y: number;
}

const PRESETS: Record<string, SpectrumPeak[]> = {
  flat: [
    { id: '1', mean: -4, std: 1.5, weight: 1 },
    { id: '2', mean: 0, std: 1.5, weight: 1 },
    { id: '3', mean: 4, std: 1.5, weight: 1 },
  ],
  illConditioned: [
    { id: '1', mean: 5, std: 0.2, weight: 0.1 },
    { id: '2', mean: -5, std: 1.0, weight: 5.0 },
  ],
  twoCluster: [
    { id: '1', mean: -2, std: 0.5, weight: 1 },
    { id: '2', mean: 2, std: 0.5, weight: 1 },
  ],
  converging: [
    { id: '1', mean: 0.1, std: 0.4, weight: 1 },
  ]
};

function generateSigmasFromPeaks(peaks: SpectrumPeak[], count = 10000): Float64Array {
  const sigmas = new Float64Array(count);
  const totalWeight = peaks.reduce((sum, p) => sum + p.weight, 0);
  let currentIndex = 0;
  for (const peak of peaks) {
    const numSamples = Math.floor((peak.weight / totalWeight) * count);
    for (let i = 0; i < numSamples && currentIndex < count; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const logVal = peak.mean + z * peak.std;
      sigmas[currentIndex++] = Math.pow(10, logVal);
    }
  }
  while (currentIndex < count) {
    sigmas[currentIndex++] = Math.pow(10, peaks[0].mean);
  }
  return sigmas.sort();
}

function generateSigmasFromSketch(
  points: SketchPoint[],
  minLog: number,
  maxLog: number,
  count = 10000
): Float64Array {
  if (points.length < 2) return new Float64Array(count).fill(Math.pow(10, (minLog + maxLog) / 2));

  const buckets = 500;
  const pdf = new Float64Array(buckets).fill(0);
  const sortedPoints = [...points].sort((a, b) => a.x - b.x);

  for (let i = 0; i < buckets; i++) {
    const xNorm = i / (buckets - 1);
    let pLeft = sortedPoints[0];
    let pRight = sortedPoints[sortedPoints.length - 1];

    for (let j = 0; j < sortedPoints.length - 1; j++) {
      if (sortedPoints[j].x <= xNorm && sortedPoints[j + 1].x >= xNorm) {
        pLeft = sortedPoints[j];
        pRight = sortedPoints[j + 1];
        break;
      }
    }

    if (pLeft === pRight) {
      pdf[i] = pLeft.y;
    } else {
      const ratio = (xNorm - pLeft.x) / (pRight.x - pLeft.x);
      pdf[i] = pLeft.y + ratio * (pRight.y - pLeft.y);
    }
    pdf[i] = Math.max(0, pdf[i]);
  }

  const cdf = new Float64Array(buckets);
  let cumulative = 0;
  for (let i = 0; i < buckets; i++) {
    cumulative += pdf[i];
    cdf[i] = cumulative;
  }
  const totalArea = cdf[buckets - 1];
  if (totalArea === 0) {
    for(let i=0; i<buckets; i++) cdf[i] = i / (buckets-1);
  } else {
    for(let i=0; i<buckets; i++) cdf[i] /= totalArea;
  }

  const sigmas = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const u = Math.random();
    let left = 0, right = buckets - 1;
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (cdf[mid] < u) left = mid + 1;
      else right = mid;
    }
    const xNorm = left / (buckets - 1);
    const logVal = minLog + xNorm * (maxLog - minLog);
    sigmas[i] = Math.pow(10, logVal);
  }
  return sigmas.sort();
}

function computeHistogram(
  values: Float64Array,
  minLog: number,
  maxLog: number,
  bins: number
): number[] {
  const histogram = new Array(bins).fill(0);
  const step = (maxLog - minLog) / bins;

  for (let i = 0; i < values.length; i++) {
    const val = values[i];
    if (val <= 0) continue;
    const logVal = Math.log10(val);
    if (logVal >= minLog && logVal < maxLog) {
      const binIdx = Math.floor((logVal - minLog) / step);
      histogram[binIdx]++;
    }
  }
  const total = values.length;
  return histogram.map(h => h / total);
}

/**
 * ------------------------------------------------------------------
 * REACT COMPONENTS
 * ------------------------------------------------------------------
 */

const usePlotly = () => {
  const [plotly, setPlotly] = useState<any>(null);
  useEffect(() => {
    if ((window as any).Plotly) {
      setPlotly((window as any).Plotly);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
    script.async = true;
    script.onload = () => setPlotly((window as any).Plotly);
    document.body.appendChild(script);
  }, []);
  return plotly;
};

const PlotlyGraph = ({ data, layout, style }: { data: any[]; layout: any; style?: any }) => {
  const Plotly = usePlotly();
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (Plotly && containerRef.current) {
      const config = { responsive: true, displayModeBar: false };
      Plotly.newPlot(containerRef.current, data, layout, config);
    }
  }, [Plotly, data, layout]);

  if (!Plotly) return <div className="flex items-center justify-center h-full bg-gray-50 text-gray-400 text-sm">Loading Visualization Lib...</div>;
  return <div ref={containerRef} style={style} className="w-full h-full" />;
};

// --- DRAWING CANVAS COMPONENT ---
const SpectrumCanvas = ({
  points,
  setPoints,
  rangeMin,
  rangeMax
}: {
  points: SketchPoint[];
  setPoints: React.Dispatch<React.SetStateAction<SketchPoint[]>>;
  rangeMin: number;
  rangeMax: number;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#f8fafc'; 
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h/2); ctx.lineTo(w, h/2);
    ctx.stroke();

    if (points.length > 0) {
      ctx.strokeStyle = '#4f46e5';
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      const sorted = [...points].sort((a, b) => a.x - b.x);
      ctx.moveTo(sorted[0].x * w, h - sorted[0].y * h);
      for (let i = 1; i < sorted.length; i++) {
        ctx.lineTo(sorted[i].x * w, h - sorted[i].y * h);
      }
      ctx.stroke();

      ctx.fillStyle = 'rgba(79, 70, 229, 0.1)';
      ctx.lineTo(sorted[sorted.length-1].x * w, h);
      ctx.lineTo(sorted[0].x * w, h);
      ctx.closePath();
      ctx.fill();
    }

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px sans-serif';
    ctx.fillText(`${rangeMin}`, 4, h - 4);
    ctx.fillText(`${rangeMax}`, w - 20, h - 4);
  }, [points, rangeMin, rangeMax]);

  const handleInteract = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    const x = (clientX - rect.left) / rect.width;
    const y = 1 - (clientY - rect.top) / rect.height;
    const clampedX = Math.max(0, Math.min(1, x));
    const clampedY = Math.max(0, Math.min(1, y));

    setPoints((prev) => {
      const newPoints = prev.filter(p => Math.abs(p.x - clampedX) > 0.02);
      return [...newPoints, { x: clampedX, y: clampedY }];
    });
  };

  return (
    <div className="relative w-full aspect-[3/2] border border-indigo-100 rounded-lg overflow-hidden shadow-inner bg-white cursor-crosshair touch-none select-none">
      <canvas
        ref={canvasRef}
        width={300}
        height={200}
        className="w-full h-full block"
        onMouseDown={(e) => { setIsDragging(true); handleInteract(e); }}
        onMouseMove={(e) => { if (isDragging) handleInteract(e); }}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onTouchStart={(e) => { setIsDragging(true); handleInteract(e); }}
        onTouchMove={(e) => { if (isDragging) handleInteract(e); }}
        onTouchEnd={() => setIsDragging(false)}
      />
      {!points.length && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-indigo-200 text-sm font-medium">
          Draw density curve
        </div>
      )}
    </div>
  );
};

export default function App() {
  const [numIters, setNumIters] = useState(8);
  const [safety, setSafety] = useState(1.01);
  const [cushion, setCushion] = useState(0.024);
  const [normalizeFrobenius, setNormalizeFrobenius] = useState(false);
  const [rangeMin, setRangeMin] = useState(-7);
  const [rangeMax, setRangeMax] = useState(7);
  const [isSketchMode, setIsSketchMode] = useState(false);
  const [peaks, setPeaks] = useState<SpectrumPeak[]>(PRESETS.twoCluster);
  const [sketchPoints, setSketchPoints] = useState<SketchPoint[]>([]);
  const [history, setHistory] = useState<Float64Array[]>([]);

  const runSimulation = useCallback(() => {
    let initialSigmas: Float64Array;
    if (isSketchMode && sketchPoints.length > 1) {
      initialSigmas = generateSigmasFromSketch(sketchPoints, rangeMin, rangeMax);
    } else {
      initialSigmas = generateSigmasFromPeaks(peaks);
    }

    if (normalizeFrobenius) {
      let sumSq = 0;
      for(let i=0; i<initialSigmas.length; i++) sumSq += initialSigmas[i] * initialSigmas[i];
      const frob = Math.sqrt(sumSq);
      if (frob > 1e-12) {
        for(let i=0; i<initialSigmas.length; i++) initialSigmas[i] /= frob;
      }
    }

    const coeffs = getCoeffsForConfig(numIters, safety, cushion);
    const results = updateSigmas(initialSigmas, coeffs);
    setHistory(results);
  }, [peaks, sketchPoints, isSketchMode, numIters, safety, cushion, rangeMin, rangeMax, normalizeFrobenius]);

  useEffect(() => {
    const timer = setTimeout(() => runSimulation(), 150);
    return () => clearTimeout(timer);
  }, [runSimulation]);

  const heatmapData = useMemo(() => {
    if (history.length === 0) return null;
    const bins = 60;
    const minLog = rangeMin;
    const maxLog = rangeMax;
    const xLabels = history.map((_, i) => i);
    const yLabels = Array.from({ length: bins }, (_, i) => minLog + i * (maxLog - minLog) / bins);

    const zData: number[][] = [];
    for (let b = 0; b < bins; b++) zData.push([]);

    history.forEach((sigmas) => {
      const density = computeHistogram(sigmas, minLog, maxLog, bins);
      density.forEach((d, binIdx) => {
        zData[binIdx].push(d);
      });
    });
    return { x: xLabels, y: yLabels, z: zData };
  }, [history, rangeMin, rangeMax]);

  const addPeak = () => {
    const center = (rangeMin + rangeMax) / 2;
    setPeaks([...peaks, { id: crypto.randomUUID(), mean: center, std: 1, weight: 1 }]);
  };
  const removePeak = (id: string) => setPeaks(peaks.filter(p => p.id !== id));
  const updatePeak = (id: string, field: keyof SpectrumPeak, value: number) => {
    setPeaks(peaks.map(p => p.id === id ? { ...p, [field]: value } : p));
  };
  const loadPreset = (key: string) => {
    setIsSketchMode(false);
    setPeaks(PRESETS[key].map(p => ({...p, id: crypto.randomUUID()})));
  };

  return (
    /* FIX: Added 'fixed inset-0' to break out of parent containers. 
      Added 'text-left' to override inherited center alignment.
    */
    <div className="fixed inset-0 w-screen h-screen bg-slate-50 text-slate-800 font-sans overflow-hidden flex flex-col text-left z-50">
      
      {/* Header */}
      <header className="flex-none h-16 bg-indigo-600 text-white px-6 flex items-center justify-between shadow-md z-20">
        <div className="flex items-center gap-3">
          <Settings className="w-6 h-6 text-indigo-200" />
          <h1 className="text-xl font-bold tracking-tight">Polar Express</h1>
        </div>
        <div className="text-xs md:text-sm text-indigo-100 opacity-90 font-medium">
          SVD Evolution Visualization
        </div>
      </header>

      {/* Main Layout Wrapper */}
      <div className="flex flex-1 flex-col md:flex-row overflow-hidden w-full">
        
        {/* LEFT SIDEBAR - Fixed width on desktop, scrollable */}
        <aside className="flex-none w-full md:w-80 bg-white border-r border-gray-200 flex flex-col overflow-y-auto z-10 text-left shadow-sm">
          
          {/* Parameters Section */}
          <div className="p-5 border-b border-gray-100">
            <h2 className="text-xs uppercase tracking-wider text-gray-500 font-bold mb-4 flex items-center gap-2">
              <RefreshCw className="w-4 h-4" /> Algorithm Params
            </h2>
            <div className="space-y-5">
              <div className="w-full">
                <div className="flex justify-between mb-1">
                  <label className="text-sm font-medium text-gray-700">Iterations</label>
                  <span className="text-sm font-mono text-indigo-600 font-bold">{numIters}</span>
                </div>
                <input type="range" min="1" max="20" step="1" value={numIters} onChange={(e) => setNumIters(parseInt(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
              </div>
              
              <div className="w-full">
                <div className="flex justify-between mb-1">
                  <label className="text-sm font-medium text-gray-700">Safety</label>
                  <span className="text-sm font-mono text-indigo-600 font-bold">{safety.toFixed(3)}</span>
                </div>
                <input type="range" min="1.00" max="1.10" step="0.001" value={safety} onChange={(e) => setSafety(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
              </div>

              <div className="w-full">
                <div className="flex justify-between mb-1">
                  <label className="text-sm font-medium text-gray-700">Cushion</label>
                  <span className="text-sm font-mono text-indigo-600 font-bold">{cushion.toFixed(3)}</span>
                </div>
                <input type="range" min="0" max="0.1" step="0.001" value={cushion} onChange={(e) => setCushion(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
              </div>

              <div className="flex items-center justify-between pt-2">
                <div className="flex items-center gap-2">
                  <Scaling className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-700 font-medium">Norm. Frobenius</span>
                </div>
                <input 
                  type="checkbox" 
                  checked={normalizeFrobenius} 
                  onChange={(e) => setNormalizeFrobenius(e.target.checked)} 
                  className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500 border-gray-300"
                />
              </div>
            </div>
          </div>

          {/* Range Settings */}
          <div className="p-5 border-b border-gray-100">
            <h2 className="text-xs uppercase tracking-wider text-gray-500 font-bold mb-4 flex items-center gap-2">
              <Maximize className="w-4 h-4" /> Grid Range (Log10)
            </h2>
            <div className="flex gap-3">
              <div className="flex-1">
                <label className="text-[10px] font-bold text-gray-400 uppercase block mb-1">Min</label>
                <input type="number" value={rangeMin} onChange={(e) => setRangeMin(Number(e.target.value))} className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-gray-50 focus:bg-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all" />
              </div>
              <div className="flex-1">
                <label className="text-[10px] font-bold text-gray-400 uppercase block mb-1">Max</label>
                <input type="number" value={rangeMax} onChange={(e) => setRangeMax(Number(e.target.value))} className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-gray-50 focus:bg-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all" />
              </div>
            </div>
          </div>

          {/* Input Mode Section */}
          <div className="p-5 flex-1">
            <h2 className="text-xs uppercase tracking-wider text-gray-500 font-bold mb-4 flex items-center gap-2">
              <Play className="w-4 h-4" /> Initial Spectrum
            </h2>
            
            <div className="flex bg-gray-100 p-1 rounded-lg mb-6">
              <button
                onClick={() => setIsSketchMode(false)}
                className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-1.5 ${!isSketchMode ? 'bg-white shadow-sm text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <Activity className="w-3.5 h-3.5" /> Parametric
              </button>
              <button
                onClick={() => setIsSketchMode(true)}
                className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-1.5 ${isSketchMode ? 'bg-white shadow-sm text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                <PenTool className="w-3.5 h-3.5" /> Sketch
              </button>
            </div>

            {isSketchMode ? (
              <div className="animate-in fade-in zoom-in duration-200">
                <p className="text-xs text-gray-500 mb-3 leading-relaxed">
                  Draw the density curve below.
                </p>
                <SpectrumCanvas points={sketchPoints} setPoints={setSketchPoints} rangeMin={rangeMin} rangeMax={rangeMax} />
                <div className="flex justify-end mt-2">
                  <button onClick={() => setSketchPoints([])} className="text-xs text-red-500 flex items-center gap-1 hover:bg-red-50 px-2 py-1 rounded font-medium transition-colors">
                    <X className="w-3 h-3" /> Clear Sketch
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-3 animate-in fade-in zoom-in duration-200">
                <div className="grid grid-cols-2 gap-2 mb-4">
                  {Object.keys(PRESETS).map(key => (
                    <button
                      key={key}
                      onClick={() => loadPreset(key)}
                      className="px-2 py-2 text-xs font-medium bg-white hover:bg-gray-50 text-gray-600 hover:text-indigo-600 rounded border border-gray-200 transition-colors capitalize shadow-sm text-left flex items-center gap-2"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-indigo-400"></div>
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </button>
                  ))}
                </div>
                
                <div className="space-y-2">
                  {peaks.map((peak, idx) => (
                    <div key={peak.id} className="bg-white p-3 rounded-lg border border-gray-200 shadow-sm group hover:border-indigo-200 transition-colors">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-bold text-indigo-900">Gaussian {idx + 1}</span>
                        <button onClick={() => removePeak(peak.id)} className="text-gray-300 hover:text-red-500 transition-colors" disabled={peaks.length <= 1}>
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <label className="text-[10px] text-gray-400 block mb-0.5">Mean</label>
                          <input type="number" step="0.1" value={peak.mean} onChange={(e) => updatePeak(peak.id, 'mean', parseFloat(e.target.value))} className="w-full px-1.5 py-1 text-xs border border-gray-200 rounded bg-gray-50 focus:bg-white focus:ring-1 focus:ring-indigo-500 outline-none" />
                        </div>
                        <div>
                          <label className="text-[10px] text-gray-400 block mb-0.5">Width</label>
                          <input type="number" step="0.1" min="0.1" value={peak.std} onChange={(e) => updatePeak(peak.id, 'std', parseFloat(e.target.value))} className="w-full px-1.5 py-1 text-xs border border-gray-200 rounded bg-gray-50 focus:bg-white focus:ring-1 focus:ring-indigo-500 outline-none" />
                        </div>
                        <div>
                          <label className="text-[10px] text-gray-400 block mb-0.5">Weight</label>
                          <input type="number" step="0.1" min="0" value={peak.weight} onChange={(e) => updatePeak(peak.id, 'weight', parseFloat(e.target.value))} className="w-full px-1.5 py-1 text-xs border border-gray-200 rounded bg-gray-50 focus:bg-white focus:ring-1 focus:ring-indigo-500 outline-none" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <button onClick={addPeak} className="mt-4 w-full py-2.5 flex items-center justify-center gap-2 text-xs font-bold text-indigo-600 border border-dashed border-indigo-200 bg-indigo-50/50 rounded-lg hover:bg-indigo-50 transition-colors">
                  <Plus className="w-3 h-3" /> Add Gaussian Peak
                </button>
              </div>
            )}
          </div>
        </aside>

        {/* RIGHT MAIN CONTENT - Takes remaining width */}
        <main className="flex-1 flex flex-col overflow-hidden bg-slate-100/50 relative">
          <div className="flex-1 overflow-y-auto p-4 md:p-6">
            
            {/* Grid Container for Charts */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 h-full content-start">
              
              {/* Chart 1: Heatmap */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-[400px] md:h-[500px] relative overflow-hidden">
                 <div className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-white">
                  <h3 className="font-bold text-gray-800 text-sm flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-500" /> Spectrum Evolution
                  </h3>
                  <div className="group relative">
                    <Info className="w-4 h-4 text-gray-300 hover:text-indigo-500 cursor-help transition-colors" />
                    <div className="absolute right-0 top-6 w-64 p-3 bg-slate-800 text-slate-100 text-xs rounded-lg shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none z-50 transition-opacity border border-slate-700">
                      <p className="font-bold mb-1">Heatmap View</p>
                      <ul className="list-disc pl-3 space-y-1 opacity-90">
                        <li>X-axis: Iteration steps</li>
                        <li>Y-axis: Log10(Singular Value)</li>
                        <li>Color: Density of values</li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="flex-1 p-2">
                  {heatmapData && (
                    <PlotlyGraph
                      data={[{ z: heatmapData.z, x: heatmapData.x, y: heatmapData.y, type: 'heatmap', colorscale: 'Viridis', showscale: false }]}
                      layout={{ margin: { t: 20, r: 20, l: 50, b: 40 }, xaxis: { title: 'Iteration' }, yaxis: { title: 'Log10(σ)' }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 11} }}
                    />
                  )}
                </div>
              </div>

              {/* Chart 2: Initial vs Final */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-[400px] md:h-[500px] relative overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-white">
                   <h3 className="font-bold text-gray-800 text-sm flex items-center gap-2">
                    <ChevronRight className="w-4 h-4 text-indigo-500" /> Initial vs. Final
                  </h3>
                </div>
                <div className="flex-1 p-2">
                  {heatmapData && history.length > 0 && (
                    <PlotlyGraph
                      data={[
                        { x: heatmapData.y, y: heatmapData.z.map(row => row[0]), type: 'scatter', mode: 'lines', name: 'Initial', fill: 'tozeroy', line: { color: '#94a3b8', width: 2 } },
                        { x: heatmapData.y, y: heatmapData.z.map(row => row[row.length - 1]), type: 'scatter', mode: 'lines', name: 'Final', fill: 'tozeroy', line: { color: '#4f46e5', width: 3 } }
                      ]}
                      layout={{ margin: { t: 20, r: 20, l: 50, b: 40 }, xaxis: { title: 'Log10(σ)' }, yaxis: { title: 'Density' }, legend: { x: 1, xanchor: 'right', y: 1 }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 11} }}
                    />
                  )}
                </div>
              </div>

              {/* Chart 3: 3D Surface (Full Width on XL) */}
              <div className="xl:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-[400px] relative overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-white">
                  <h3 className="font-bold text-gray-800 text-sm flex items-center gap-2">
                    <Maximize className="w-4 h-4 text-indigo-500" /> 3D Density Landscape
                  </h3>
                  <span className="text-[10px] uppercase font-bold text-gray-400 bg-gray-100 px-2 py-0.5 rounded">Interactive</span>
                </div>
                <div className="flex-1 p-2">
                  {heatmapData && (
                    <PlotlyGraph
                      data={[{ z: heatmapData.z.map(row => row), x: heatmapData.x, y: heatmapData.y, type: 'surface', colorscale: 'Viridis', contours: { z: { show: true, usecolormap: true, highlightcolor: "#42f462", project: { z: true } } } }]}
                      layout={{ margin: { t: 0, r: 0, l: 0, b: 0 }, scene: { xaxis: { title: 'Iter' }, yaxis: { title: 'Log10(σ)' }, zaxis: { title: 'Density' }, camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } } }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 11} }}
                    />
                  )}
                </div>
              </div>

            </div>
          </div>
        </main>

      </div>
    </div>
  );
}