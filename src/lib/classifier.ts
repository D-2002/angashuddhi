/**
 * AngaShuddhi — ML Classifier
 *
 * Loads the trained Random Forest (exported as ONNX with zipmap=False)
 * and runs inference entirely in the browser via onnxruntime-web.
 *
 * zipmap=False is required because onnxruntime-web WASM cannot read
 * sequence-of-maps outputs (the skl2onnx default). With zipmap=False,
 * probabilities are a flat float32 tensor of shape [1, n_classes].
 */

import * as ort from 'onnxruntime-web';
import type { AramandiFeatures } from './features';

// ─── Constants (must match Python training script exactly) ─────────────────

export const FEATURE_COLS = [
  'left_knee_angle',
  'right_knee_angle',
  'avg_knee_angle',
  'sitting_depth',
  'torso_lean',
  'left_knee_dev',
  'right_knee_dev',
  'knee_bend_delta',
  'hip_level_delta',
  'com_lateral_offset',
] as const;

// Must match CLASS_ORDER in training script exactly
export const CLASS_NAMES = [
  'correct',
  'forward_lean',
  'insufficient_depth',
  'knee_collapse',
  'left_knee_collapse',
  'right_knee_collapse',
  'weight_shift',
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

export interface ClassifierResult {
  predictedClass:  ClassName;
  confidence:      number;
  probabilities:   Record<ClassName, number>;
  feedbackHeadline: string;
  feedbackDetail:   string;
}

// ─── Scaler ────────────────────────────────────────────────────────────────

interface ScalerParams {
  feature_names: string[];
  class_names:   string[];
  mean:          number[];
  std:           number[];
}

let scalerParams: ScalerParams | null = null;

async function loadScaler(): Promise<ScalerParams> {
  if (scalerParams) return scalerParams;
  const res    = await fetch('/model/angashuddhi_scaler.json');
  scalerParams = (await res.json()) as ScalerParams;
  return scalerParams;
}

function scaleFeatures(raw: number[], params: ScalerParams): Float32Array {
  const scaled = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    scaled[i] = (raw[i] - params.mean[i]) / (params.std[i] || 1);
  }
  return scaled;
}

// ─── ONNX Session ──────────────────────────────────────────────────────────

let session:   ort.InferenceSession | null = null;
let isLoading  = false;

export async function loadClassifier(): Promise<void> {
  if (session || isLoading) return;
  isLoading = true;
  try {
    ort.env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    session = await ort.InferenceSession.create('/model/angashuddhi_model.onnx', {
      executionProviders: ['wasm'],
    });
    // Log output names on first load so we can verify format in console
    console.log('[AngaShuddhi] ONNX model loaded. Output names:', session.outputNames);
  } catch (err) {
    console.error('[AngaShuddhi] Failed to load ONNX model:', err);
  } finally {
    isLoading = false;
  }
}

// ─── Inference ─────────────────────────────────────────────────────────────

export async function classifyPose(
  features: AramandiFeatures,
): Promise<ClassifierResult | null> {
  if (!session) {
    await loadClassifier();
    if (!session) return null;
  }

  const scaler  = await loadScaler();
  const n       = FEATURE_COLS.length;

  // Build feature vector in exact training order
  const raw: number[] = [
    features.leftKneeAngle,
    features.rightKneeAngle,
    features.avgKneeAngle,
    features.sittingDepthRatio,
    features.torsoLeanAngle,
    features.leftKneeDeviation,
    features.rightKneeDeviation,
    features.kneeBendDelta,
    features.hipLevelDelta,
    features.comLateralOffset,
  ];

  const scaled = scaleFeatures(raw, scaler);
  const tensor = new ort.Tensor('float32', scaled, [1, n]);
  const feeds  = { float_input: tensor };

  let results: ort.InferenceSession.OnnxValueMapType;
  try {
    results = await session.run(feeds);
  } catch (err) {
    console.error('[AngaShuddhi] ONNX inference error:', err);
    return null;
  }

  // ── Parse label output ──────────────────────────────────────────────────
  // output_label: int64 tensor, shape [1]
  const labelOutput = results['label'];
  if (!labelOutput) {
    console.error('[AngaShuddhi] label missing. Available:', Object.keys(results));
    return null;
  }
  const predictedIdx   = Number(labelOutput.data[0]);
  const predictedClass = CLASS_NAMES[predictedIdx] as ClassName;

  // ── Parse probability output ────────────────────────────────────────────
  // With zipmap=False: output_probability is float32 tensor, shape [1, 7]
  // Data is a flat Float32Array: [p_class0, p_class1, ..., p_class6]
  
  const probOutput = results['probabilities'];
  if (!probOutput) {
    console.error('[AngaShuddhi] probabilities missing. Available:', Object.keys(results));
    return null;
  }

  // Flat float32 array — one probability per class in CLASS_NAMES order
  const probData    = probOutput.data as Float32Array;
  const probabilities = Object.fromEntries(
    CLASS_NAMES.map((name, i) => [name, probData[i] ?? 0])
  ) as Record<ClassName, number>;

  const confidence = probabilities[predictedClass];

  return {
    predictedClass,
    confidence,
    probabilities,
    feedbackHeadline: HEADLINES[predictedClass],
    feedbackDetail:   DETAILS[predictedClass],
  };
}

// ─── Feedback copy ─────────────────────────────────────────────────────────

const HEADLINES: Record<ClassName, string> = {
  correct:             'Aramandi looks correct',
  forward_lean:        'Forward lean detected',
  insufficient_depth:  'Insufficient sitting depth',
  knee_collapse:       'Both knees collapsing inward',
  left_knee_collapse:  'Left knee collapsing inward',
  right_knee_collapse: 'Right knee collapsing inward',
  weight_shift:        'Weight shift detected',
};

const DETAILS: Record<ClassName, string> = {
  correct:
    'Good posture — maintain this form and focus on holding stability.',
  forward_lean:
    'Your torso is tilting forward. Lift your chest, engage your core, and keep your spine vertical over your hips.',
  insufficient_depth:
    'You are not sitting deep enough. Bend your knees further until your thighs approach parallel to the floor.',
  knee_collapse:
    'Both knees are falling inward. Widen your foot stance first, then actively rotate your hips outward to push both knees out.',
  left_knee_collapse:
    'Your left knee is falling inward. Push it outward — the rotation comes from your left hip, not just the knee.',
  right_knee_collapse:
    'Your right knee is falling inward. Push it outward — the rotation comes from your right hip, not just the knee.',
  weight_shift:
    'Your weight is shifting sideways. Root both feet evenly into the floor and centre your hips over the midpoint.',
};