import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Play, RefreshCw, Settings, Plus, Trash2, Info, Maximize, PenTool, Activity, X, Scaling, ChevronRight, LayoutTemplate, Eye, Menu, Columns, Monitor } from 'lucide-react';

// --- Types & Interfaces ---

interface Peak {
  id: string;
  mean: number;
  std: number;
  weight: number;
}

interface Point {
  x: number;
  y: number;
}

interface HeatmapData {
  x: number[];
  y: number[];
  z: number[][];
  plotMin: number;
  plotMax: number;
}

interface ExpandedViewData {
  type: 'heatmap' | 'line' | 'surface';
  data: HeatmapData;
  title: string;
}

interface Histories {
  polar: Float64Array[];
  newton: Float64Array[];
  jordan: Float64Array[];
}

// Extend Window interface for Plotly
declare global {
  interface Window {
    Plotly: any;
  }
}

/**
 * ------------------------------------------------------------------
 * CORE ALGORITHMS
 * ------------------------------------------------------------------
 */

// Polar Express Coefficients
const DEFAULT_RAW_COEFFS = [
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
  safety = DEFAULT_SAFETY,
  cushion = DEFAULT_CUSHION
) {
  let scaled = DEFAULT_RAW_COEFFS.map((triple, i) => {
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

function updateSigmas(
  sigmas: Float64Array | number[], 
  algorithm: string,
  coeffs: number[][], // For Polar
  numIters: number    // For Newton/Jordan
) {
  const history = [new Float64Array(sigmas)];
  let currentSigmas = new Float64Array(sigmas);

  if (algorithm === 'polar') {
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
  } else if (algorithm === 'newton') {
    // Newton-Schulz: x_k+1 = 0.5 * x_k * (3 - x_k^2)
    for (let k = 0; k < numIters; k++) {
      const nextSigmas = new Float64Array(currentSigmas.length);
      for (let i = 0; i < currentSigmas.length; i++) {
        const s = currentSigmas[i];
        nextSigmas[i] = 0.5 * s * (3 - s * s);
      }
      currentSigmas = nextSigmas;
      history.push(currentSigmas);
    }
  } else if (algorithm === 'jordan') {
    // Jordan (Matrix Sign / Newton Square Root): x_k+1 = 0.5 * (x_k + 1/x_k)
    for (let k = 0; k < numIters; k++) {
      const nextSigmas = new Float64Array(currentSigmas.length);
      for (let i = 0; i < currentSigmas.length; i++) {
        const s = currentSigmas[i];
        // Prevent division by zero
        if (Math.abs(s) < 1e-15) nextSigmas[i] = s; 
        else nextSigmas[i] = 0.5 * (s + 1.0 / s);
      }
      currentSigmas = nextSigmas;
      history.push(currentSigmas);
    }
  }

  return history;
}

/**
 * ------------------------------------------------------------------
 * DATA GENERATION & HELPERS
 * ------------------------------------------------------------------
 */

const PRESETS: Record<string, Peak[]> = {
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

function generateSigmasFromPeaks(peaks: Peak[], count = 10000) {
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
  points: Point[],
  minLog: number,
  maxLog: number,
  count = 10000
) {
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
  values: Float64Array | number[],
  minLog: number,
  maxLog: number,
  bins: number
) {
  const histogram = new Array(bins).fill(0);
  // Avoid division by zero if min == max
  const safeMax = maxLog === minLog ? maxLog + 1 : maxLog;
  const step = (safeMax - minLog) / bins;

  for (let i = 0; i < values.length; i++) {
    const val = values[i];
    if (val <= 0) continue; // Skip negative or zero sigmas (diverged)
    const logVal = Math.log10(val);
    if (logVal >= minLog && logVal < safeMax) {
      const binIdx = Math.floor((logVal - minLog) / step);
      if (binIdx >= 0 && binIdx < bins) {
        histogram[binIdx]++;
      }
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
    if (window.Plotly) {
      setPlotly(window.Plotly);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
    script.async = true;
    script.onload = () => setPlotly(window.Plotly);
    document.body.appendChild(script);
  }, []);
  return plotly;
};

interface PlotlyGraphProps {
  data: any[];
  layout: any;
  style?: React.CSSProperties;
  config?: any;
}

// Optimization: Wrapped in React.memo to prevent unnecessary re-renders
const PlotlyGraph = React.memo(({ data, layout, style, config }: PlotlyGraphProps) => {
  const Plotly = usePlotly();
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (Plotly && containerRef.current) {
      const defaultConfig = { responsive: true, displayModeBar: false };
      // Optimization: Use Plotly.react (faster update) instead of newPlot (destroy/create)
      Plotly.react(containerRef.current, data, layout, { ...defaultConfig, ...config });
    }
  }, [Plotly, data, layout, config]);

  if (!Plotly) return <div className="flex items-center justify-center h-full bg-gray-50 text-gray-400 text-sm">Loading...</div>;
  return <div ref={containerRef} style={style} className="w-full h-full" />;
});
PlotlyGraph.displayName = 'PlotlyGraph';

// --- DRAWING CANVAS COMPONENT ---
interface SpectrumCanvasProps {
  points: Point[];
  setPoints: React.Dispatch<React.SetStateAction<Point[]>>;
  rangeMin: number;
  rangeMax: number;
}

const SpectrumCanvas = ({
  points,
  setPoints,
  rangeMin,
  rangeMax
}: SpectrumCanvasProps) => {
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

// --- EXTRACTED AND MEMOIZED COMPONENT CARDS ---
// Moving these outside App() prevents them from being redefined (and unmounting/remounting) on every render

interface CardProps {
    title: string;
    data: HeatmapData | null;
    heightClass?: string;
    showInfo?: boolean;
    onExpand?: () => void;
    className?: string;
}

const HeatmapCard = React.memo(({ title, data, heightClass = "h-[400px]", showInfo = true, onExpand }: CardProps) => (
  <div className={`bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col ${heightClass} relative overflow-hidden group hover:border-indigo-300 transition-all`}>
    <div 
      className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-white cursor-pointer hover:bg-gray-50 transition-colors"
      onClick={onExpand}
    >
      <h3 className="font-bold text-gray-800 text-sm flex items-center gap-2">
        <Activity className="w-4 h-4 text-indigo-500" /> {title}
      </h3>
      <div className="flex items-center gap-2">
          <button className="p-1 rounded hover:bg-gray-200 text-gray-400 hover:text-indigo-600 transition-colors" title="Expand">
            <Maximize className="w-3.5 h-3.5" />
          </button>
          {showInfo && (
          <div className="group/info relative" onClick={(e) => e.stopPropagation()}>
            <Info className="w-4 h-4 text-gray-300 hover:text-indigo-500 cursor-help transition-colors" />
            <div className="absolute right-0 top-6 w-64 p-3 bg-slate-800 text-slate-100 text-xs rounded-lg shadow-xl opacity-0 group-hover/info:opacity-100 pointer-events-none z-50 transition-opacity border border-slate-700">
              <p className="font-bold mb-1">Heatmap View</p>
              <ul className="list-disc pl-3 space-y-1 opacity-90">
                <li>X-axis: Iteration steps</li>
                <li>Y-axis: Log10(Singular Value)</li>
                <li>Color: Density of values</li>
              </ul>
            </div>
          </div>
          )}
      </div>
    </div>
    <div className="flex-1 p-2 relative">
      {data && (
        <PlotlyGraph
          data={[{ z: data.z, x: data.x, y: data.y, type: 'heatmap', colorscale: 'Viridis', showscale: false }]}
          layout={{ margin: { t: 10, r: 10, l: 40, b: 30 }, xaxis: { title: 'Iter' }, yaxis: { title: 'Log10(σ)', range: [data.plotMin, data.plotMax] }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 10} }}
        />
      )}
    </div>
  </div>
));
HeatmapCard.displayName = 'HeatmapCard';

const LineCard = React.memo(({ title, data, heightClass = "h-[400px]", onExpand }: CardProps) => (
  <div className={`bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col ${heightClass} relative overflow-hidden group hover:border-indigo-300 transition-all`}>
    <div 
      className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-white cursor-pointer hover:bg-gray-50 transition-colors"
      onClick={onExpand}
    >
        <h3 className="font-bold text-gray-800 text-sm flex items-center gap-2">
        <ChevronRight className="w-4 h-4 text-indigo-500" /> {title}
      </h3>
      <button className="p-1 rounded hover:bg-gray-200 text-gray-400 hover:text-indigo-600 transition-colors" title="Expand">
          <Maximize className="w-3.5 h-3.5" />
      </button>
    </div>
    <div className="flex-1 p-2 relative">
      {data && (
        <PlotlyGraph
          data={[
            { x: data.y, y: data.z.map(row => row[0]), type: 'scatter', mode: 'lines', name: 'Initial', fill: 'tozeroy', line: { color: '#94a3b8', width: 2 } },
            { x: data.y, y: data.z.map(row => row[row.length - 1]), type: 'scatter', mode: 'lines', name: 'Final', fill: 'tozeroy', line: { color: '#4f46e5', width: 3 } }
          ]}
          layout={{ margin: { t: 10, r: 10, l: 40, b: 30 }, xaxis: { title: 'Log10(σ)', range: [data.plotMin, data.plotMax] }, yaxis: { title: 'Density' }, showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 10} }}
        />
      )}
    </div>
  </div>
));
LineCard.displayName = 'LineCard';

const SurfaceCard = React.memo(({ title, data, heightClass = "h-[400px]", className = "", onExpand }: CardProps) => (
  <div className={`bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col ${heightClass} ${className} relative overflow-hidden group hover:border-indigo-300 transition-all`}>
    <div 
      className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-white cursor-pointer hover:bg-gray-50 transition-colors"
      onClick={onExpand}
    >
      <h3 className="font-bold text-gray-800 text-sm flex items-center gap-2">
        <Maximize className="w-4 h-4 text-indigo-500" /> {title}
      </h3>
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase font-bold text-gray-400 bg-gray-100 px-2 py-0.5 rounded">Interactive</span>
        <button className="p-1 rounded hover:bg-gray-200 text-gray-400 hover:text-indigo-600 transition-colors" title="Expand">
          <Maximize className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
    <div className="flex-1 p-2 relative">
      {data && (
        <PlotlyGraph
          data={[{ z: data.z.map(row => row), x: data.x, y: data.y, type: 'surface', colorscale: 'Viridis', contours: { z: { show: true, usecolormap: true, highlightcolor: "#42f462", project: { z: true } } } }]}
          layout={{ margin: { t: 0, r: 0, l: 0, b: 0 }, scene: { xaxis: { title: 'Iter' }, yaxis: { title: 'Log10(σ)', range: [data.plotMin, data.plotMax] }, zaxis: { title: 'Density' }, camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } } }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 11} }}
        />
      )}
    </div>
  </div>
));
SurfaceCard.displayName = 'SurfaceCard';


export default function App() {
  // -- State: UI --
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [viewMode, setViewMode] = useState<'single' | 'compare'>('single'); 
  const [expandedView, setExpandedView] = useState<ExpandedViewData | null>(null);

  // -- State: Parameters --
  const [algorithm, setAlgorithm] = useState<string>('polar');
  const [numIters, setNumIters] = useState(8);
  const [safety, setSafety] = useState(1.01);
  const [cushion, setCushion] = useState(0.024);
  const [normalizeFrobenius, setNormalizeFrobenius] = useState(false);
  
  // -- State: Range --
  const [rangeMin, setRangeMin] = useState(-7);
  const [rangeMax, setRangeMax] = useState(7);
  const [showFullRange, setShowFullRange] = useState(false);

  // -- State: Spectrum Mode --
  const [isSketchMode, setIsSketchMode] = useState(false);

  // -- State: Data --
  const [peaks, setPeaks] = useState<Peak[]>(PRESETS.twoCluster);
  const [sketchPoints, setSketchPoints] = useState<Point[]>([]);
  
  // Store histories for all algorithms
  const [histories, setHistories] = useState<Histories>({
    polar: [],
    newton: [],
    jordan: []
  });

  // -- Run Simulation --
  const runSimulation = useCallback(() => {
    let initialSigmas;
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

    // Run all three algorithms regardless of view mode to support instant switching
    const polarCoeffs = getCoeffsForConfig(numIters, safety, cushion);
    
    const hPolar = updateSigmas(initialSigmas, 'polar', polarCoeffs, numIters);
    const hNewton = updateSigmas(initialSigmas, 'newton', [], numIters);
    const hJordan = updateSigmas(initialSigmas, 'jordan', [], numIters);

    setHistories({
      polar: hPolar,
      newton: hNewton,
      jordan: hJordan
    });
  }, [peaks, sketchPoints, isSketchMode, numIters, safety, cushion, rangeMin, rangeMax, normalizeFrobenius]);

  // Run on mount and when params change
  useEffect(() => {
    const timer = setTimeout(() => runSimulation(), 150);
    return () => clearTimeout(timer);
  }, [runSimulation]);

  // -- Visualization Data Helper --
  const generateHeatmapData = useCallback((historyData: Float64Array[]) : HeatmapData | null => {
    if (!historyData || historyData.length === 0) return null;
    const bins = 60;
    let plotMin = rangeMin;
    let plotMax = rangeMax;

    if (showFullRange) {
      let globalMin = Infinity;
      let globalMax = -Infinity;
      for (const stepData of historyData) {
        for (const val of stepData) {
          if (val > 1e-20 && val < 1e20) {
             const lv = Math.log10(val);
             if (lv < globalMin) globalMin = lv;
             if (lv > globalMax) globalMax = lv;
          }
        }
      }
      if (globalMin === Infinity) { globalMin = -5; globalMax = 5; }
      plotMin = globalMin - 0.5;
      plotMax = globalMax + 0.5;
    }

    const xLabels = historyData.map((_, i) => i);
    const yLabels = Array.from({ length: bins }, (_, i) => plotMin + i * (plotMax - plotMin) / bins);

    const zData: number[][] = [];
    for (let b = 0; b < bins; b++) zData.push([]);

    historyData.forEach((sigmas) => {
      const density = computeHistogram(sigmas, plotMin, plotMax, bins);
      density.forEach((d, binIdx) => {
        zData[binIdx].push(d);
      });
    });

    return { x: xLabels, y: yLabels, z: zData, plotMin, plotMax };
  }, [rangeMin, rangeMax, showFullRange]);

  // Compute heatmaps
  const heatmaps: Record<string, HeatmapData | null> = useMemo(() => {
    return {
      polar: generateHeatmapData(histories.polar),
      newton: generateHeatmapData(histories.newton),
      jordan: generateHeatmapData(histories.jordan),
    };
  }, [histories, generateHeatmapData]);

  // -- Handlers --
  const addPeak = () => {
    const center = (rangeMin + rangeMax) / 2;
    setPeaks([...peaks, { id: crypto.randomUUID(), mean: center, std: 1, weight: 1 }]);
  };
  const removePeak = (id: string) => setPeaks(peaks.filter(p => p.id !== id));
  const updatePeak = (id: string, field: keyof Peak, value: number) => {
    setPeaks(peaks.map(p => p.id === id ? { ...p, [field]: value } : p));
  };
  const loadPreset = (key: string) => {
    setIsSketchMode(false);
    if (PRESETS[key]) {
        setPeaks(PRESETS[key].map(p => ({...p, id: crypto.randomUUID()})));
    }
  };
  const handleRangeChange = (valStr: string, setter: (v: number) => void) => {
    if (valStr === '' || valStr === '-') {
      setter(0); 
      return;
    }
    const val = parseFloat(valStr);
    if (!isNaN(val)) {
      setter(val);
    }
  };

  // Define expand handlers with useCallback to keep their identity stable
  const handleExpand = useCallback((type: 'heatmap' | 'line' | 'surface', alg: string, title: string) => {
    const data = heatmaps[alg];
    if (data) {
      setExpandedView({ type, data, title });
    }
  }, [heatmaps]);

  return (
    <div className="fixed inset-0 w-screen h-screen bg-slate-50 text-slate-800 font-sans overflow-hidden flex flex-col text-left z-50">
      
      {/* Header */}
      <header className="flex-none h-16 bg-indigo-600 text-white px-4 md:px-6 flex items-center justify-between shadow-md z-20">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 hover:bg-indigo-700 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            <Menu className="w-5 h-5 text-indigo-100" />
          </button>
          <div className="flex items-center gap-3">
            <Settings className="w-6 h-6 text-indigo-200" />
            <h1 className="text-xl font-bold tracking-tight">Polar Express</h1>
          </div>
        </div>
        <div className="text-xs md:text-sm text-indigo-100 opacity-90 font-medium flex items-center gap-4">
           <span className="hidden md:inline">SVD Visualization</span>
        </div>
      </header>

      {/* Main Layout Wrapper */}
      <div className="flex flex-1 overflow-hidden w-full relative">
        
        {/* LEFT SIDEBAR */}
        <aside 
          className={`
            flex-none bg-white border-r border-gray-200 flex flex-col overflow-y-auto z-10 text-left shadow-sm transition-all duration-300 ease-in-out
            ${isSidebarOpen ? 'w-full md:w-80 translate-x-0' : 'w-0 -translate-x-full overflow-hidden'}
          `}
        >
          {/* View Mode */}
          <div className="p-5 border-b border-gray-100">
             <h2 className="text-xs uppercase tracking-wider text-gray-500 font-bold mb-4 flex items-center gap-2">
               <Monitor className="w-4 h-4" /> View Mode
             </h2>
             <div className="flex bg-gray-100 p-1 rounded-lg">
               <button
                 onClick={() => setViewMode('single')}
                 className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-1.5 ${viewMode === 'single' ? 'bg-white shadow-sm text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
               >
                 <LayoutTemplate className="w-3.5 h-3.5" /> Single
               </button>
               <button
                 onClick={() => setViewMode('compare')}
                 className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-1.5 ${viewMode === 'compare' ? 'bg-white shadow-sm text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
               >
                 <Columns className="w-3.5 h-3.5" /> Compare
               </button>
             </div>
          </div>

          {/* Algorithm Selection (Only in Single Mode) */}
           <div className={`p-5 border-b border-gray-100 transition-all duration-200 ${viewMode === 'compare' ? 'opacity-50 pointer-events-none grayscale' : ''}`}>
            <h2 className="text-xs uppercase tracking-wider text-gray-500 font-bold mb-4 flex items-center gap-2">
              <LayoutTemplate className="w-4 h-4" /> Algorithm
            </h2>
            <select 
              value={algorithm} 
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-indigo-500 outline-none"
            >
              <option value="polar">Polar Express (New)</option>
              <option value="newton">Newton-Schulz</option>
              <option value="jordan">Jordan (Matrix Sign)</option>
            </select>
          </div>

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
              
              {/* Polar Params - Only enabled if single & polar OR if compare mode (since Polar is part of comparison) */}
              <div className={`transition-all duration-200 ${(viewMode === 'single' && algorithm !== 'polar') ? 'opacity-30 pointer-events-none' : ''}`}>
                  <div className="w-full mb-5">
                    <div className="flex justify-between mb-1">
                      <label className="text-sm font-medium text-gray-700">PE Safety</label>
                      <span className="text-sm font-mono text-indigo-600 font-bold">{safety.toFixed(3)}</span>
                    </div>
                    <input type="range" min="1.00" max="1.10" step="0.001" value={safety} onChange={(e) => setSafety(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                  </div>

                  <div className="w-full">
                    <div className="flex justify-between mb-1">
                      <label className="text-sm font-medium text-gray-700">PE Cushion</label>
                      <span className="text-sm font-mono text-indigo-600 font-bold">{cushion.toFixed(3)}</span>
                    </div>
                    <input type="range" min="0" max="0.1" step="0.001" value={cushion} onChange={(e) => setCushion(parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                  </div>
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
            
            <div className="flex items-center gap-2 mb-4 bg-indigo-50 p-2 rounded-lg border border-indigo-100">
              <Eye className="w-4 h-4 text-indigo-600" />
              <span className="text-xs font-bold text-indigo-700 flex-1">Show Full Range</span>
              <input 
                type="checkbox" 
                checked={showFullRange} 
                onChange={(e) => setShowFullRange(e.target.checked)} 
                className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500 border-gray-300"
              />
            </div>

            <div className={`flex gap-3 transition-opacity ${showFullRange ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
              <div className="flex-1">
                <label className="text-[10px] font-bold text-gray-400 uppercase block mb-1">Min</label>
                <input 
                  type="number" 
                  value={rangeMin} 
                  onChange={(e) => handleRangeChange(e.target.value, setRangeMin)} 
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-gray-50 focus:bg-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all" 
                />
              </div>
              <div className="flex-1">
                <label className="text-[10px] font-bold text-gray-400 uppercase block mb-1">Max</label>
                <input 
                  type="number" 
                  value={rangeMax} 
                  onChange={(e) => handleRangeChange(e.target.value, setRangeMax)} 
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-gray-50 focus:bg-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all" 
                />
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

        {/* RIGHT MAIN CONTENT */}
        <main className="flex-1 flex flex-col overflow-hidden bg-slate-100/50 relative w-full">
          <div className="flex-1 overflow-y-auto p-4 md:p-6 pb-32">
            
            {viewMode === 'single' ? (
              // --- SINGLE VIEW ---
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 h-full content-start">
                <HeatmapCard 
                    title="Spectrum Evolution" 
                    data={heatmaps[algorithm]} 
                    heightClass="h-[400px] md:h-[500px]"
                    onExpand={() => handleExpand('heatmap', algorithm, 'Spectrum Evolution')}
                />
                <LineCard 
                    title="Initial vs. Final" 
                    data={heatmaps[algorithm]} 
                    heightClass="h-[400px] md:h-[500px]" 
                    onExpand={() => handleExpand('line', algorithm, 'Initial vs Final')}
                />

                {/* 3D Surface (Full Width on XL) */}
                <SurfaceCard 
                    title="3D Density Landscape" 
                    data={heatmaps[algorithm]} 
                    className="xl:col-span-2" 
                    heightClass="h-[400px]" 
                    onExpand={() => handleExpand('surface', algorithm, '3D Density Landscape')}
                />
              </div>
            ) : (
              // --- COMPARISON VIEW ---
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-full content-start">
                {/* Column 1: Polar */}
                <div className="flex flex-col gap-4">
                   <div className="font-bold text-center text-indigo-900 bg-indigo-50 py-2 rounded-lg border border-indigo-100">Polar Express</div>
                   <HeatmapCard 
                        title="Polar Evolution" 
                        data={heatmaps.polar} 
                        heightClass="h-[300px]" 
                        showInfo={false} 
                        onExpand={() => handleExpand('heatmap', 'polar', 'Polar Evolution')}
                   />
                   <LineCard 
                        title="Polar Initial vs Final" 
                        data={heatmaps.polar} 
                        heightClass="h-[250px]" 
                        onExpand={() => handleExpand('line', 'polar', 'Polar Initial vs Final')}
                   />
                   <SurfaceCard 
                        title="Polar 3D" 
                        data={heatmaps.polar} 
                        heightClass="h-[250px]" 
                        onExpand={() => handleExpand('surface', 'polar', 'Polar 3D')}
                   />
                </div>

                {/* Column 2: Newton */}
                <div className="flex flex-col gap-4">
                   <div className="font-bold text-center text-indigo-900 bg-indigo-50 py-2 rounded-lg border border-indigo-100">Newton-Schulz</div>
                   <HeatmapCard 
                        title="Newton Evolution" 
                        data={heatmaps.newton} 
                        heightClass="h-[300px]" 
                        showInfo={false} 
                        onExpand={() => handleExpand('heatmap', 'newton', 'Newton Evolution')}
                   />
                   <LineCard 
                        title="Newton Initial vs Final" 
                        data={heatmaps.newton} 
                        heightClass="h-[250px]" 
                        onExpand={() => handleExpand('line', 'newton', 'Newton Initial vs Final')}
                   />
                   <SurfaceCard 
                        title="Newton 3D" 
                        data={heatmaps.newton} 
                        heightClass="h-[250px]" 
                        onExpand={() => handleExpand('surface', 'newton', 'Newton 3D')}
                   />
                </div>

                {/* Column 3: Jordan */}
                <div className="flex flex-col gap-4">
                   <div className="font-bold text-center text-indigo-900 bg-indigo-50 py-2 rounded-lg border border-indigo-100">Jordan (Matrix Sign)</div>
                   <HeatmapCard 
                        title="Jordan Evolution" 
                        data={heatmaps.jordan} 
                        heightClass="h-[300px]" 
                        showInfo={false} 
                        onExpand={() => handleExpand('heatmap', 'jordan', 'Jordan Evolution')}
                   />
                   <LineCard 
                        title="Jordan Initial vs Final" 
                        data={heatmaps.jordan} 
                        heightClass="h-[250px]" 
                        onExpand={() => handleExpand('line', 'jordan', 'Jordan Initial vs Final')}
                   />
                   <SurfaceCard 
                        title="Jordan 3D" 
                        data={heatmaps.jordan} 
                        heightClass="h-[250px]" 
                        onExpand={() => handleExpand('surface', 'jordan', 'Jordan 3D')}
                   />
                </div>
              </div>
            )}

          </div>
        </main>

      </div>

      {/* EXPANDED VIEW OVERLAY */}
      {expandedView && (
        <div 
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-8 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200"
            onClick={() => setExpandedView(null)}
        >
            <div 
                className="bg-white w-full h-full max-w-6xl max-h-[85vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in zoom-in-95 duration-200 ring-1 ring-white/10"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-white">
                    <div className="flex items-center gap-3">
                        {expandedView.type === 'heatmap' && <Activity className="w-6 h-6 text-indigo-500" />}
                        {expandedView.type === 'line' && <ChevronRight className="w-6 h-6 text-indigo-500" />}
                        {expandedView.type === 'surface' && <Maximize className="w-6 h-6 text-indigo-500" />}
                        <h2 className="text-xl font-bold text-gray-800">{expandedView.title}</h2>
                    </div>
                    <button 
                        onClick={() => setExpandedView(null)}
                        className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-500 hover:text-red-500"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>
                <div className="flex-1 p-4 bg-gray-50/50">
                    {expandedView.type === 'heatmap' && (
                        <PlotlyGraph
                            data={[{ z: expandedView.data.z, x: expandedView.data.x, y: expandedView.data.y, type: 'heatmap', colorscale: 'Viridis', showscale: true }]}
                            layout={{ margin: { t: 20, r: 20, l: 60, b: 50 }, xaxis: { title: 'Iteration' }, yaxis: { title: 'Log10(σ)', range: [expandedView.data.plotMin, expandedView.data.plotMax] }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 14} }}
                        />
                    )}
                    {expandedView.type === 'line' && (
                        <PlotlyGraph
                            data={[
                            { x: expandedView.data.y, y: expandedView.data.z.map(row => row[0]), type: 'scatter', mode: 'lines', name: 'Initial', fill: 'tozeroy', line: { color: '#94a3b8', width: 3 } },
                            { x: expandedView.data.y, y: expandedView.data.z.map(row => row[row.length - 1]), type: 'scatter', mode: 'lines', name: 'Final', fill: 'tozeroy', line: { color: '#4f46e5', width: 4 } }
                            ]}
                            layout={{ margin: { t: 20, r: 20, l: 60, b: 50 }, xaxis: { title: 'Log10(σ)', range: [expandedView.data.plotMin, expandedView.data.plotMax] }, yaxis: { title: 'Density' }, showlegend: true, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 14} }}
                        />
                    )}
                    {expandedView.type === 'surface' && (
                        <PlotlyGraph
                            data={[{ z: expandedView.data.z.map(row => row), x: expandedView.data.x, y: expandedView.data.y, type: 'surface', colorscale: 'Viridis', contours: { z: { show: true, usecolormap: true, highlightcolor: "#42f462", project: { z: true } } } }]}
                            layout={{ margin: { t: 0, r: 0, l: 0, b: 0 }, scene: { xaxis: { title: 'Iter' }, yaxis: { title: 'Log10(σ)', range: [expandedView.data.plotMin, expandedView.data.plotMax] }, zaxis: { title: 'Density' }, camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } } }, paper_bgcolor: 'rgba(0,0,0,0)', font: {family: 'sans-serif', size: 14} }}
                        />
                    )}
                </div>
                <div className="bg-gray-50 px-6 py-3 border-t border-gray-200 text-right text-xs text-gray-400">
                    Click outside or press Esc to close
                </div>
            </div>
        </div>
      )}
    </div>
  );
}