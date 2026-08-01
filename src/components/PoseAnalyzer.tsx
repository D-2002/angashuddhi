'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import {
  extractAramandiFeatures,
  scoreAramandiFeatures,
  type MetricScore,
  type Landmark,
} from '@/lib/features';
import {
  loadClassifier,
  classifyPose,
  type ClassifierResult,
} from '@/lib/classifier';
import {
  runPriorityEngine,
  detectRecurringErrors,
  computeSessionTrend,
  type ReasoningResult,
} from '@/lib/reasoning/priorityEngine';
import {
  resolveConflict,
  getTeacherFeedback,
  type ConflictResolution,
  type TeacherFeedback,
} from '@/lib/reasoning/conflictResolver';

const WASM_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// ── History buffers — kept in refs so they don't trigger re-renders ───────────
const MAX_HISTORY = 120;   // ~12 seconds of history at 10fps updates

const statusBorder: Record<string, string> = {
  good:  'border-green-700', warn: 'border-amber-700', error: 'border-red-700',
};
const statusBg: Record<string, string> = {
  good:  'bg-green-950/60', warn: 'bg-amber-950/60', error: 'bg-red-950/60',
};
const statusText: Record<string, string> = {
  good:  'text-green-400', warn: 'text-amber-400', error: 'text-red-400',
};
const statusIcon: Record<string, string> = { good: '✓', warn: '⚠', error: '✗' };

const tierColour: Record<string, string> = {
  fatal:       'text-red-400',
  structural:  'text-amber-400',
  refinement:  'text-blue-400',
  none:        'text-green-400',
};

export default function PoseAnalyzer() {
  const videoRef        = useRef<HTMLVideoElement>(null);
  const canvasRef       = useRef<HTMLCanvasElement>(null);
  const landmarkerRef   = useRef<PoseLandmarker | null>(null);
  const rafRef          = useRef<number>(0);
  const lastTimeRef     = useRef<number>(-1);
  const streamRef       = useRef<MediaStream | null>(null);
  const scoreUpdateRef  = useRef<number>(0);
  const sessionStartRef = useRef<number>(Date.now());
  const frameCountRef   = useRef<number>(0);

  // History buffers — not React state, never trigger re-renders
  const errorHistoryRef  = useRef<string[][]>([]);
  const scoreHistoryRef  = useRef<number[]>([]);

  const [modelStatus,   setModelStatus]   = useState<'loading' | 'ready' | 'error'>('loading');
  const [mlReady,       setMlReady]       = useState(false);
  const [hasSource,     setHasSource]     = useState(false);
  const [poseCount,     setPoseCount]     = useState(0);
  const [fps,           setFps]           = useState(0);

  // Analysis state
  const [ruleScores,    setRuleScores]    = useState<Record<string, MetricScore>>({});
  const [mlResult,      setMlResult]      = useState<ClassifierResult | null>(null);
  const [reasoning,     setReasoning]     = useState<ReasoningResult | null>(null);
  const [conflict,      setConflict]      = useState<ConflictResolution | null>(null);

  // On-demand teacher feedback
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [teacherFeedback, setTeacherFeedback] = useState<TeacherFeedback | null>(null);

  const fpsRef = useRef<{ frames: number; last: number }>({ frames: 0, last: performance.now() });

  // ── Init ────────────────────────────────────────────────────────────────────
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
      } catch {
        setModelStatus('error');
      }
    })();
    loadClassifier().then(() => setMlReady(true));
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  // ── Detection loop ──────────────────────────────────────────────────────────
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

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw; canvas.height = vh;
    }

    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(video, 0, 0, vw, vh);

    const results = landmarker.detectForVideo(video, performance.now());

    const du = new DrawingUtils(ctx);
    for (const lm of results.landmarks) {
      du.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF9A', lineWidth: 2 });
      du.drawLandmarks(lm, { color: '#FF6B6B', fillColor: '#FF6B6B', lineWidth: 1, radius: 4 });
    }

    if (results.landmarks.length > 0) {
      const lm = results.landmarks[0] as Landmark[];
      const KEY_LOWER = [23, 24, 25, 26, 27, 28];
      const lowerVisible = KEY_LOWER.every(i => (lm[i].visibility ?? 0) > 0.3);

      if (lowerVisible) {
        const features    = extractAramandiFeatures(lm, vw, vh);
        const now         = performance.now();

        if (now - scoreUpdateRef.current > 150) {
          scoreUpdateRef.current = now;
          frameCountRef.current++;

          const scores = scoreAramandiFeatures(features);
          setRuleScores(scores);

          // ML classification (async)
          if (mlReady) {
            classifyPose(features).then(ml => {
              if (!ml) return;
              setMlResult(ml);

              // ── Reasoning layer ─────────────────────────────────────────
              const reason = runPriorityEngine(
                scores,
                features,
                errorHistoryRef.current,
                scoreHistoryRef.current,
              );
              const conf = resolveConflict(scores, ml);

              setReasoning(reason);
              setConflict(conf);

              // Update history buffers
              const frameErrors = reason.allErrors.map(e => e.errorKey);
              errorHistoryRef.current = [
                ...errorHistoryRef.current.slice(-(MAX_HISTORY - 1)),
                frameErrors,
              ];
              scoreHistoryRef.current = [
                ...scoreHistoryRef.current.slice(-(MAX_HISTORY - 1)),
                reason.overallScore,
              ];
            });
          }
        }
      } else {
        setRuleScores({});
        setMlResult(null);
        setReasoning(null);
      }
    } else {
      setRuleScores({});
      setMlResult(null);
      setReasoning(null);
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
  }, [mlReady]);

  // ── On-demand teacher feedback ──────────────────────────────────────────────
  const requestTeacherFeedback = async () => {
    if (!reasoning || !mlResult || !conflict) return;
    setFeedbackLoading(true);
    setTeacherFeedback(null);
    try {
      const fb = await getTeacherFeedback({
        reasoning,
        mlResult,
        conflict,
        sessionDuration: (Date.now() - sessionStartRef.current) / 1000,
        frameCount:      frameCountRef.current,
      });
      setTeacherFeedback(fb);
    } finally {
      setFeedbackLoading(false);
    }
  };

  // ── Source handlers ─────────────────────────────────────────────────────────
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    stopSource();
    const video = videoRef.current!;
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => { setHasSource(true); video.play(); rafRef.current = requestAnimationFrame(runDetectionLoop); };
  };

  const handleWebcam = async () => {
    stopSource();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'user' } });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.src = ''; video.srcObject = stream;
      video.onloadeddata = () => { setHasSource(true); video.play(); rafRef.current = requestAnimationFrame(runDetectionLoop); };
    } catch { alert('Could not access webcam — check browser permissions.'); }
  };

  const stopSource = () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null; lastTimeRef.current = -1;
    setHasSource(false); setPoseCount(0); setFps(0);
    setRuleScores({}); setMlResult(null); setReasoning(null);
    setTeacherFeedback(null);
    errorHistoryRef.current = []; scoreHistoryRef.current = [];
    sessionStartRef.current = Date.now(); frameCountRef.current = 0;
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  };

  const hasScores     = Object.keys(ruleScores).length > 0;
  const hasReasoning  = reasoning !== null;

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-6">
      <div className="w-full max-w-3xl">

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-medium tracking-tight">
            AngaShuddhi <span className="text-gray-500 font-normal">अंगशुद्धि</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">Bringing computational precision to classical dance training</p>
        </div>

        {/* Status */}
        <div className="flex items-center gap-4 mb-5 text-sm flex-wrap">
          {modelStatus === 'loading' && <span className="text-amber-400 animate-pulse">⏳ Loading pose model…</span>}
          {modelStatus === 'ready'   && <span className="text-green-400">✓ Pose model ready</span>}
          {modelStatus === 'error'   && <span className="text-red-400">✗ Pose model failed</span>}
          <span className={mlReady ? 'text-green-400' : 'text-gray-500'}>{mlReady ? '✓ Classifier ready' : '⏳ Loading classifier…'}</span>
          {hasSource && <><span className={poseCount > 0 ? 'text-green-300' : 'text-amber-300'}>{poseCount > 0 ? `✓ ${poseCount} pose` : '⚠ No pose'}</span><span className="text-gray-500">{fps} fps</span></>}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-3 mb-5">
          <label className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition cursor-pointer ${modelStatus === 'ready' ? 'bg-indigo-600 hover:bg-indigo-500' : 'bg-gray-700 opacity-40 pointer-events-none'}`}>
            Upload video
            <input type="file" accept="video/*" className="hidden" onChange={handleVideoUpload} disabled={modelStatus !== 'ready'} />
          </label>
          <button onClick={handleWebcam} disabled={modelStatus !== 'ready'} className={`px-4 py-2 rounded-lg text-sm font-medium transition ${modelStatus === 'ready' ? 'bg-gray-700 hover:bg-gray-600 cursor-pointer' : 'bg-gray-700 opacity-40 cursor-not-allowed'}`}>Use webcam</button>
          {hasSource && <button onClick={stopSource} className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-800 hover:bg-gray-700 cursor-pointer transition">Stop</button>}
        </div>

        {/* Canvas */}
        <div className="w-full bg-gray-900 rounded-xl overflow-hidden relative min-h-48">
          <video ref={videoRef} className="hidden" playsInline muted loop />
          <canvas ref={canvasRef} className="w-full h-auto block" />
          {!hasSource && <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">Upload a video or start webcam — face the camera directly</div>}
        </div>

        {/* ── PRIMARY FEEDBACK PANEL (Reasoning layer) ─────────────────────── */}
        {hasSource && hasReasoning && reasoning && (
          <div className="mt-5 p-4 bg-gray-900 rounded-xl border border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-medium text-gray-300">
                Teacher Feedback
                <span className="ml-2 text-xs text-gray-500 font-normal">Pedagogical priority engine</span>
              </h2>
              <div className="flex items-center gap-2">
                {/* Score */}
                <span className={`text-sm font-mono font-bold ${reasoning.overallScore >= 80 ? 'text-green-400' : reasoning.overallScore >= 60 ? 'text-amber-400' : 'text-red-400'}`}>
                  {reasoning.overallScore}/100
                </span>
                {/* Session trend */}
                {reasoning.sessionTrend !== 'stable' && (
                  <span className={`text-xs ${reasoning.sessionTrend === 'improving' ? 'text-green-400' : 'text-red-400'}`}>
                    {reasoning.sessionTrend === 'improving' ? '↑ improving' : '↓ deteriorating'}
                  </span>
                )}
              </div>
            </div>

            {/* Primary correction — the most important thing right now */}
            {reasoning.topError ? (
              <div className="mb-3">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className={`text-xs font-medium uppercase tracking-wide ${tierColour[reasoning.topError.tier]}`}>
                    {reasoning.topError.tier} error
                  </span>
                  {reasoning.topError.isRecurring && (
                    <span className="text-xs text-red-400 font-medium">● recurring</span>
                  )}
                </div>
                <p className="text-white text-sm font-medium leading-relaxed mb-1">
                  {reasoning.phraseToDisplay}
                </p>
                {reasoning.topError.rootCause !== 'Direct technique error' && (
                  <p className="text-xs text-gray-500 leading-relaxed">
                    Root cause: {reasoning.topError.rootCause}
                  </p>
                )}
              </div>
            ) : (
              <p className="text-green-400 text-sm font-medium mb-3">{reasoning.phraseToDisplay}</p>
            )}

            {/* Conflict note if interesting */}
            {conflict?.displayNote && (
              <p className="text-xs text-indigo-400 mb-3 italic">{conflict.displayNote}</p>
            )}

            {/* Secondary errors — listed but not expanded */}
            {reasoning.allErrors.length > 1 && (
              <div className="mt-2 pt-2 border-t border-gray-800">
                <p className="text-xs text-gray-500 mb-1.5">Also detected:</p>
                <div className="flex flex-wrap gap-1.5">
                  {reasoning.allErrors.slice(1).map(err => (
                    <span key={err.errorKey} className={`text-xs px-2 py-0.5 rounded-full border ${err.tier === 'fatal' ? 'border-red-800 text-red-400' : err.tier === 'structural' ? 'border-amber-800 text-amber-400' : 'border-gray-700 text-gray-400'}`}>
                      {err.errorKey.replace(/([A-Z])/g, ' $1').toLowerCase().trim()}
                      {err.isRecurring ? ' ●' : ''}
                    </span>
                  ))}
                </div>
                <p className="text-xs text-gray-600 mt-1.5">Fix the primary error first — addressing multiple errors at once is less effective.</p>
              </div>
            )}

            {/* Get teacher feedback button */}
            <div className="mt-4 pt-3 border-t border-gray-800">
              <button
                onClick={requestTeacherFeedback}
                disabled={feedbackLoading || !hasReasoning}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition w-full ${feedbackLoading ? 'bg-gray-700 text-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-500 cursor-pointer'}`}
              >
                {feedbackLoading ? '⏳ Getting teacher feedback…' : 'Get detailed teacher feedback'}
              </button>
              <p className="text-xs text-gray-600 mt-1.5 text-center">Analyses your full session — takes ~3 seconds</p>
            </div>
          </div>
        )}

        {/* ── TEACHER FEEDBACK MODAL ────────────────────────────────────────── */}
        {teacherFeedback && (
          <div className="mt-4 p-4 bg-indigo-950/40 border border-indigo-800 rounded-xl">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-medium text-indigo-300">Teacher Feedback</h2>
              <button onClick={() => setTeacherFeedback(null)} className="text-gray-500 hover:text-gray-300 text-xs cursor-pointer">dismiss</button>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-xs text-indigo-400 mb-1 font-medium">Primary correction</p>
                <p className="text-sm text-white leading-relaxed">{teacherFeedback.primaryCorrection}</p>
              </div>
              <div>
                <p className="text-xs text-indigo-400 mb-1 font-medium">Why it matters</p>
                <p className="text-sm text-gray-300 leading-relaxed">{teacherFeedback.whyItMatters}</p>
              </div>
              <div>
                <p className="text-xs text-indigo-400 mb-1 font-medium">What to do next</p>
                <p className="text-sm text-gray-300 leading-relaxed">{teacherFeedback.whatToDoNext}</p>
              </div>
              <div className="pt-2 border-t border-indigo-900">
                <p className="text-xs text-gray-500 leading-relaxed italic">{teacherFeedback.sessionSummary}</p>
              </div>
            </div>
          </div>
        )}

        {/* ── BIOMECHANICAL METRICS (Rule engine — secondary display) ────────── */}
        {hasSource && (
          <details className="mt-4">
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300 transition mb-3">
              Biomechanical metrics (rule engine)
            </summary>
            {hasScores ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-3">
                {Object.entries(ruleScores).map(([key, metric]) => (
                  <div key={key} className={`p-3 rounded-xl border ${statusBorder[metric.status]} ${statusBg[metric.status]}`}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs text-gray-400">{metric.label}</span>
                      <span className={`text-xs font-bold ${statusText[metric.status]}`}>{statusIcon[metric.status]}</span>
                    </div>
                    <div className="text-base font-mono font-semibold text-white mb-1.5">{metric.displayValue}</div>
                    <div className={`text-xs leading-tight ${statusText[metric.status]}`}>{metric.feedback}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-600 mt-2">No pose detected.</p>
            )}
          </details>
        )}

      </div>
    </div>
  );
}
