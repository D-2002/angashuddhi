'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

// Pin WASM version to match npm package
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// MediaPipe 33-landmark indices — reference for Day 2 feature extraction
// 0=nose, 11=L.shoulder, 12=R.shoulder, 13=L.elbow, 14=R.elbow
// 23=L.hip, 24=R.hip, 25=L.knee, 26=R.knee
// 27=L.ankle, 28=R.ankle, 29=L.heel, 30=R.heel
// 31=L.foot-index, 32=R.foot-index

export default function PoseAnalyzer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(-1);
  const streamRef = useRef<MediaStream | null>(null);

  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [hasSource, setHasSource] = useState(false);
  const [poseCount, setPoseCount] = useState(0);
  const [fps, setFps] = useState(0);
  const fpsCounterRef = useRef<{ frames: number; last: number }>({ frames: 0, last: performance.now() });

  // ─── Init MediaPipe (runs once on mount) ─────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: 'GPU', // falls back to CPU automatically if GPU unavailable
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
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

  // ─── Core detection loop ──────────────────────────────────────────────────
  const runDetectionLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker) return;
    if (video.paused || video.ended) return;

    // Skip if video frame hasn't changed (avoids duplicate inference)
    if (video.currentTime === lastTimeRef.current) {
      rafRef.current = requestAnimationFrame(runDetectionLoop);
      return;
    }
    lastTimeRef.current = video.currentTime;

    // Always sync canvas to video's natural resolution
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw;
      canvas.height = vh;
    }

    const ctx = canvas.getContext('2d')!;

    // Draw video frame — canvas is the display surface, video element stays hidden
    ctx.drawImage(video, 0, 0, vw, vh);

    // Run MediaPipe inference
    const results = landmarker.detectForVideo(video, performance.now());

    // Draw skeleton overlay
    const drawingUtils = new DrawingUtils(ctx);
    for (const landmarks of results.landmarks) {
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: '#00FF9A',
        lineWidth: 2,
      });
      drawingUtils.drawLandmarks(landmarks, {
        color: '#FF6B6B',
        fillColor: '#FF6B6B',
        lineWidth: 1,
        radius: 4,
      });
    }

    // ── Day 2 prep: log Aramandi-relevant landmarks ──
    // Comment this out once you're extracting features properly
    if (results.landmarks.length > 0) {
      const lm = results.landmarks[0];
      // Normalized [0,1] coords relative to image. z = depth relative to hip midpoint.
      console.log('[AngaShuddhi] Aramandi landmarks:', {
        leftHip:    { x: +lm[23].x.toFixed(3), y: +lm[23].y.toFixed(3), z: +lm[23].z.toFixed(3) },
        rightHip:   { x: +lm[24].x.toFixed(3), y: +lm[24].y.toFixed(3), z: +lm[24].z.toFixed(3) },
        leftKnee:   { x: +lm[25].x.toFixed(3), y: +lm[25].y.toFixed(3), z: +lm[25].z.toFixed(3) },
        rightKnee:  { x: +lm[26].x.toFixed(3), y: +lm[26].y.toFixed(3), z: +lm[26].z.toFixed(3) },
        leftAnkle:  { x: +lm[27].x.toFixed(3), y: +lm[27].y.toFixed(3) },
        rightAnkle: { x: +lm[28].x.toFixed(3), y: +lm[28].y.toFixed(3) },
        leftHeel:   { x: +lm[29].x.toFixed(3), y: +lm[29].y.toFixed(3) },
        rightHeel:  { x: +lm[30].x.toFixed(3), y: +lm[30].y.toFixed(3) },
        leftFoot:   { x: +lm[31].x.toFixed(3), y: +lm[31].y.toFixed(3) },
        rightFoot:  { x: +lm[32].x.toFixed(3), y: +lm[32].y.toFixed(3) },
      });
    }

    // FPS counter
    const now = performance.now();
    fpsCounterRef.current.frames++;
    if (now - fpsCounterRef.current.last >= 1000) {
      setFps(fpsCounterRef.current.frames);
      fpsCounterRef.current = { frames: 0, last: now };
    }

    setPoseCount(results.landmarks.length);
    rafRef.current = requestAnimationFrame(runDetectionLoop);
  }, []);

  // ─── Source: video file ───────────────────────────────────────────────────
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

  // ─── Source: webcam ───────────────────────────────────────────────────────
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
      console.error('[AngaShuddhi] Webcam access denied:', err);
      alert('Could not access webcam. Check browser permissions.');
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

    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-6">
      <div className="w-full max-w-3xl">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-medium tracking-tight">
            AngaShuddhi <span className="text-gray-500 font-normal">अंगशुद्धि</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Bringing computational precision to classical dance training
          </p>
        </div>

        {/* Status bar */}
        <div className="flex items-center gap-4 mb-6 text-sm">
          {modelStatus === 'loading' && (
            <span className="text-amber-400 animate-pulse">⏳ Loading MediaPipe model (~10 MB)…</span>
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

        {/* Controls */}
        <div className="flex flex-wrap gap-3 mb-6">
          <label
            className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${
              modelStatus === 'ready'
                ? 'bg-indigo-600 hover:bg-indigo-500'
                : 'bg-gray-700 opacity-40 cursor-not-allowed pointer-events-none'
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

        {/* Canvas — this IS the display surface, video element stays hidden */}
        <div className="w-full bg-gray-900 rounded-xl overflow-hidden relative min-h-48">
          <video ref={videoRef} className="hidden" playsInline muted loop />
          <canvas ref={canvasRef} className="w-full h-auto block" />
          {!hasSource && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
              Upload a video or start webcam to begin
            </div>
          )}
        </div>

        {/* Landmark reference — you'll use this heavily on Day 2 */}
        <div className="mt-6 p-4 bg-gray-900 rounded-xl">
          <p className="text-xs font-medium text-gray-300 mb-3">
            MediaPipe landmark indices (open DevTools → Console to see live values)
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-1 text-xs text-gray-400 font-mono">
            <span>0 → nose</span>
            <span>11, 12 → shoulders</span>
            <span>13, 14 → elbows</span>
            <span>23, 24 → hips</span>
            <span>25, 26 → knees</span>
            <span>27, 28 → ankles</span>
            <span>29, 30 → heels</span>
            <span>31, 32 → foot index</span>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Coordinates are normalized [0,1]. x increases right, y increases down, z = depth (relative to hip midpoint).
          </p>
        </div>

      </div>
    </div>
  );
}