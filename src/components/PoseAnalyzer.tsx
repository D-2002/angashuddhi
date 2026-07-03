'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import {
  extractAramandiFeatures,
  scoreAramandiFeatures,
  type MetricScore,
  type Landmark,
} from '@/lib/features';

const WASM_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const statusBorder: Record<string, string> = {
  good:  'border-green-700',
  warn:  'border-amber-700',
  error: 'border-red-700',
};
const statusBg: Record<string, string> = {
  good:  'bg-green-950/60',
  warn:  'bg-amber-950/60',
  error: 'bg-red-950/60',
};
const statusText: Record<string, string> = {
  good:  'text-green-400',
  warn:  'text-amber-400',
  error: 'text-red-400',
};
const statusIcon: Record<string, string> = {
  good: '✓',
  warn: '⚠',
  error: '✗',
};

export default function PoseAnalyzer() {
  const videoRef       = useRef<HTMLVideoElement>(null);
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const landmarkerRef  = useRef<PoseLandmarker | null>(null);
  const rafRef         = useRef<number>(0);
  const lastTimeRef    = useRef<number>(-1);
  const streamRef      = useRef<MediaStream | null>(null);
  const scoreUpdateRef = useRef<number>(0);

  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [hasSource,   setHasSource]   = useState(false);
  const [poseCount,   setPoseCount]   = useState(0);
  const [fps,         setFps]         = useState(0);
  const [scores,      setScores]      = useState<Record<string, MetricScore>>({});

  const fpsRef = useRef<{ frames: number; last: number }>({ frames: 0, last: performance.now() });

  // ── Init MediaPipe ─────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence:  0.5,
          minTrackingConfidence:      0.5,
        });
        setModelStatus('ready');
      } catch (err) {
        console.error('[AngaShuddhi] MediaPipe init failed:', err);
        setModelStatus('error');
      }
    })();
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ── Detection loop ─────────────────────────────────────────────────────────
  const runDetectionLoop = useCallback(() => {
    const video      = videoRef.current;
    const canvas     = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker) return;
    if (video.paused || video.ended) return;

    if (video.currentTime === lastTimeRef.current) {
      rafRef.current = requestAnimationFrame(runDetectionLoop);
      return;
    }
    lastTimeRef.current = video.currentTime;

    const vw = video.videoWidth  || 640;
    const vh = video.videoHeight || 480;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width  = vw;
      canvas.height = vh;
    }

    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(video, 0, 0, vw, vh);

    const results = landmarker.detectForVideo(video, performance.now());

    // Draw skeleton
    const drawingUtils = new DrawingUtils(ctx);
    for (const landmarks of results.landmarks) {
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: '#00FF9A', lineWidth: 2,
      });
      drawingUtils.drawLandmarks(landmarks, {
        color: '#FF6B6B', fillColor: '#FF6B6B', lineWidth: 1, radius: 4,
      });
    }

    if (results.landmarks.length > 0) {
      const lm = results.landmarks[0] as Landmark[];

      // ── DIAGNOSTIC: log actual visibility scores every 2 seconds ──────────
      // Read these in DevTools → Console while standing in Aramandi
      // We use this to set the real threshold — remove after calibration
      const KEY_LOWER = [23, 24, 25, 26, 27, 28];
      const now0 = performance.now();
      if (now0 - scoreUpdateRef.current > 2000) {
        console.log('[AngaShuddhi] Lower body visibility scores:',
          KEY_LOWER.map(i => ({
            landmark: ['L.Hip','R.Hip','L.Knee','R.Knee','L.Ankle','R.Ankle'][KEY_LOWER.indexOf(i)],
            id: i,
            visibility: lm[i].visibility?.toFixed(3) ?? 'undefined',
          }))
        );
      }

      // ── Visibility filter — TEMPORARILY set to 0.3 so cards still show ───
      // We will tighten this once we read your real values above
      const VISIBILITY_THRESHOLD = 0.3;
      const lowerVisible = KEY_LOWER.every(
        i => (lm[i].visibility ?? 0) > VISIBILITY_THRESHOLD
      );

      if (lowerVisible) {
        const features = extractAramandiFeatures(lm, vw, vh);

        // ── DIAGNOSTIC: log raw feature values in Aramandi ────────────────
        // Stand in full Aramandi and read these numbers
        // We use this to calibrate thresholds in features.ts
        const now1 = performance.now();
        if (now1 - scoreUpdateRef.current > 2000) {
          console.table({
            sittingDepthRatio:   +features.sittingDepthRatio.toFixed(3),
            avgKneeAngle:        +features.avgKneeAngle.toFixed(1),
            leftKneeAngle:       +features.leftKneeAngle.toFixed(1),
            rightKneeAngle:      +features.rightKneeAngle.toFixed(1),
            torsoLeanAngle:      +features.torsoLeanAngle.toFixed(1),
            leftKneeDeviation:   +features.leftKneeDeviation.toFixed(3),
            rightKneeDeviation:  +features.rightKneeDeviation.toFixed(3),
            kneeBendDelta:       +features.kneeBendDelta.toFixed(1),
            hipLevelDelta:       +features.hipLevelDelta.toFixed(3),
            comLateralOffset:    +features.comLateralOffset.toFixed(3),
          });
          scoreUpdateRef.current = now1;
        }

        const now2 = performance.now();
        if (now2 - scoreUpdateRef.current > 100) {
          setScores(scoreAramandiFeatures(features));
          scoreUpdateRef.current = now2;
        }

      } else {
        // Log which landmarks failed so we know what's being blocked
        console.log('[AngaShuddhi] Visibility check failed for:',
          KEY_LOWER
            .filter(i => (lm[i].visibility ?? 0) <= VISIBILITY_THRESHOLD)
            .map(i => ['L.Hip','R.Hip','L.Knee','R.Knee','L.Ankle','R.Ankle'][KEY_LOWER.indexOf(i)])
        );
        setScores({});
      }
    } else {
      setScores({});
    }

    // FPS
    const now = performance.now();
    fpsRef.current.frames++;
    if (now - fpsRef.current.last >= 1000) {
      setFps(fpsRef.current.frames);
      fpsRef.current = { frames: 0, last: now };
    }

    setPoseCount(results.landmarks.length);
    rafRef.current = requestAnimationFrame(runDetectionLoop);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Source handlers ────────────────────────────────────────────────────────
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    stopSource();
    const video = videoRef.current!;
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => {
      setHasSource(true);
      video.play();
      rafRef.current = requestAnimationFrame(runDetectionLoop);
    };
  };

  const handleWebcam = async () => {
    stopSource();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
      });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.src = '';
      video.srcObject = stream;
      video.onloadeddata = () => {
        setHasSource(true);
        video.play();
        rafRef.current = requestAnimationFrame(runDetectionLoop);
      };
    } catch (err) {
      console.error('[AngaShuddhi] Webcam denied:', err);
      alert('Could not access webcam — check browser permissions.');
    }
  };

  const stopSource = () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    lastTimeRef.current = -1;
    setHasSource(false);
    setPoseCount(0);
    setFps(0);
    setScores({});
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  };

  const hasScores = Object.keys(scores).length > 0;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-6">
      <div className="w-full max-w-3xl">

        <div className="mb-6">
          <h1 className="text-2xl font-medium tracking-tight">
            AngaShuddhi{' '}
            <span className="text-gray-500 font-normal">अंगशुद्धि</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Bringing computational precision to classical dance training
          </p>
        </div>

        <div className="flex items-center gap-4 mb-5 text-sm flex-wrap">
          {modelStatus === 'loading' && (
            <span className="text-amber-400 animate-pulse">⏳ Loading model…</span>
          )}
          {modelStatus === 'ready' && (
            <span className="text-green-400">✓ Model ready</span>
          )}
          {modelStatus === 'error' && (
            <span className="text-red-400">✗ Model load failed — see console</span>
          )}
          {hasSource && (
            <>
              <span className={poseCount > 0 ? 'text-green-300' : 'text-amber-300'}>
                {poseCount > 0 ? `✓ ${poseCount} pose detected` : '⚠ No pose in frame'}
              </span>
              <span className="text-gray-500">{fps} fps</span>
            </>
          )}
        </div>

        <div className="flex flex-wrap gap-3 mb-5">
          <label
            className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${
              modelStatus === 'ready'
                ? 'bg-indigo-600 hover:bg-indigo-500'
                : 'bg-gray-700 opacity-40 pointer-events-none'
            }`}
          >
            Upload video
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleVideoUpload}
              disabled={modelStatus !== 'ready'}
            />
          </label>

          <button
            onClick={handleWebcam}
            disabled={modelStatus !== 'ready'}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              modelStatus === 'ready'
                ? 'bg-gray-700 hover:bg-gray-600 cursor-pointer'
                : 'bg-gray-700 opacity-40 cursor-not-allowed'
            }`}
          >
            Use webcam
          </button>

          {hasSource && (
            <button
              onClick={stopSource}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 hover:bg-gray-700 cursor-pointer transition"
            >
              Stop
            </button>
          )}
        </div>

        <div className="w-full bg-gray-900 rounded-xl overflow-hidden relative min-h-48">
          <video ref={videoRef} className="hidden" playsInline muted loop />
          <canvas ref={canvasRef} className="w-full h-auto block" />
          {!hasSource && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
              Upload a video or start webcam to begin
            </div>
          )}
        </div>

        {hasSource && (
          <div className="mt-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-medium text-gray-300">
                Aramandi Analysis
              </h2>
              {!hasScores && poseCount === 0 && (
                <span className="text-xs text-gray-500">
                  Stand in frame — ensure full body is visible
                </span>
              )}
              {!hasScores && poseCount > 0 && (
                <span className="text-xs text-amber-500">
                  Pose detected but visibility too low — check console
                </span>
              )}
            </div>

            {hasScores ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Object.entries(scores).map(([key, metric]) => (
                  <div
                    key={key}
                    className={`p-3 rounded-xl border ${statusBorder[metric.status]} ${statusBg[metric.status]}`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs text-gray-400 leading-tight">
                        {metric.label}
                      </span>
                      <span className={`text-xs font-bold ${statusText[metric.status]}`}>
                        {statusIcon[metric.status]}
                      </span>
                    </div>
                    <div className="text-base font-mono font-semibold text-white mb-1.5">
                      {metric.displayValue}
                    </div>
                    <div className={`text-xs leading-tight ${statusText[metric.status]}`}>
                      {metric.feedback}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Array.from({ length: 7 }).map((_, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl border border-gray-800 bg-gray-900/50 animate-pulse h-24"
                  />
                ))}
              </div>
            )}
          </div>
        )}

        <details className="mt-6">
          <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-400 transition">
            Landmark index reference
          </summary>
          <div className="mt-2 p-4 bg-gray-900 rounded-xl grid grid-cols-2 sm:grid-cols-3 gap-y-1 text-xs text-gray-500 font-mono">
            <span>0  → nose</span>
            <span>11, 12 → shoulders</span>
            <span>13, 14 → elbows</span>
            <span>23, 24 → hips</span>
            <span>25, 26 → knees</span>
            <span>27, 28 → ankles</span>
            <span>29, 30 → heels</span>
            <span>31, 32 → foot index</span>
          </div>
        </details>

      </div>
    </div>
  );
}