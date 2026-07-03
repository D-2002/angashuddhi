/**
 * AngaShuddhi — Feature Extraction Layer
 * Calibrated against real Aramandi data (July 2026 session).
 *
 * Key references:
 * [1] Escamilla et al. (2001). Knee biomechanics of the dynamic squat exercise.
 *     Medicine & Science in Sports & Exercise, 33(1), 127–141.
 * [2] Sigward & Powers (2006). The influence of gender on knee kinematics.
 *     Clinical Biomechanics, 21(1), 41–48.
 *     NOTE: FPPA from [2] is intentionally NOT used for Aramandi knee tracking —
 *     FPPA assumes parallel foot stance. Aramandi requires ~45° external rotation
 *     per foot, making the ankle laterally displaced by design. Using FPPA would
 *     incorrectly flag correct Aramandi as knee valgus collapse.
 *     Instead we use the Bharatanatyam-specific rule: knee must be LATERAL to hip.
 * [3] Coplan (2002). Ballet dancer's turnout and its relationship to injury.
 *     JOSPT, 32(11), 579–584.
 * [4] De Leva (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia
 *     parameters. Journal of Biomechanics, 29(9), 1223–1230.
 */

export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface AramandiFeatures {
  /**
   * Hip–Knee–Ankle angle in degrees (pixel-space, aspect-ratio corrected).
   * 180° = straight leg. ~90–110° = good Aramandi depth.
   * Ref: Escamilla 2001.
   */
  leftKneeAngle: number;
  rightKneeAngle: number;
  avgKneeAngle: number;

  /**
   * How deeply the dancer is sitting.
   * 0 = standing (hips at shoulder level), 1 = hips at ankle level (maximum).
   * Calibrated: real full Aramandi ≈ 0.49–0.52 on a frontal camera.
   * Target: > 0.44
   */
  sittingDepthRatio: number;

  /**
   * Torso lean from vertical in degrees.
   * Lateral lean: from X,Y vectors (reliable).
   * Forward lean: from MediaPipe Z depth (noisier, directionally informative).
   * Combined: max of both signals.
   * Target: < 10°
   */
  torsoLeanAngle: number;

  /**
   * Bharatanatyam-specific knee tracking.
   * Measures how far each knee is LATERAL to its hip landmark.
   *
   * Positive = knee is wider than hip (correct — knees opening outward).
   * Negative = knee is narrower than hip (incorrect — knee collapsing inward).
   *
   * Formula: leftKneeDeviation  = leftHip.x  − leftKnee.x
   *          rightKneeDeviation = rightKnee.x − rightHip.x
   *
   * In MediaPipe coords, x increases rightward. So:
   *   Left knee correct  → leftKnee.x  < leftHip.x  → deviation > 0
   *   Right knee correct → rightKnee.x > rightHip.x → deviation > 0
   *
   * This directly encodes the teacher rule "open your knees outward."
   * Replaces FPPA (Sigward & Powers 2006) which assumes parallel foot stance
   * and is inappropriate for Aramandi's ~45° external rotation.
   */
  leftKneeDeviation: number;
  rightKneeDeviation: number;

  /**
   * |leftKneeAngle − rightKneeAngle| in degrees.
   * Left–right bend symmetry. Target: < 10°.
   */
  kneeBendDelta: number;

  /**
   * |leftHip.y − rightHip.y| in normalized units.
   * Hip tilt / weight shift. Target: < 0.025.
   */
  hipLevelDelta: number;

  /**
   * |hipMidpoint.x − footMidpoint.x| in normalized units.
   * Approximate lateral CoM offset. Ref: De Leva 1996.
   * Target: < 0.04.
   */
  comLateralOffset: number;
}

export interface MetricScore {
  rawValue: number;
  displayValue: string;
  label: string;
  status: 'good' | 'warn' | 'error';
  feedback: string;
}

// ─── Geometry helpers ──────────────────────────────────────────────────────────

type P2 = { x: number; y: number };

function angleDeg(a: P2, b: P2, c: P2): number {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const dot = ba.x * bc.x + ba.y * bc.y;
  const mag =
    Math.sqrt(ba.x ** 2 + ba.y ** 2) *
    Math.sqrt(bc.x ** 2 + bc.y ** 2);
  if (mag < 1e-9) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

function toPx(lm: Landmark, w: number, h: number): P2 {
  return { x: lm.x * w, y: lm.y * h };
}

function midLm(a: Landmark, b: Landmark): Landmark {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

// ─── Feature extraction ────────────────────────────────────────────────────────

export function extractAramandiFeatures(
  lm: Landmark[],
  w: number,
  h: number
): AramandiFeatures {
  // Named references — MediaPipe 33-point body model
  const lShoulder = lm[11], rShoulder = lm[12];
  const lHip      = lm[23], rHip      = lm[24];
  const lKnee     = lm[25], rKnee     = lm[26];
  const lAnkle    = lm[27], rAnkle    = lm[28];

  // ── 1. Knee flexion angles (pixel-space for aspect-ratio correctness) ──────
  const leftKneeAngle  = angleDeg(toPx(lHip, w, h),  toPx(lKnee, w, h),  toPx(lAnkle, w, h));
  const rightKneeAngle = angleDeg(toPx(rHip, w, h),  toPx(rKnee, w, h),  toPx(rAnkle, w, h));
  const avgKneeAngle   = (leftKneeAngle + rightKneeAngle) / 2;

  // ── 2. Sitting depth ratio ────────────────────────────────────────────────
  // Pure vertical comparison in normalised y — no aspect ratio issue here.
  const hipMid      = midLm(lHip, rHip);
  const shoulderMid = midLm(lShoulder, rShoulder);
  const ankleMidY   = (lAnkle.y + rAnkle.y) / 2;
  const totalHeight = ankleMidY - shoulderMid.y;
  const hipDescent  = hipMid.y - shoulderMid.y;
  const sittingDepthRatio = totalHeight > 0.05
    ? Math.min(1, Math.max(0, hipDescent / totalHeight))
    : 0;

  // ── 3. Torso lean ─────────────────────────────────────────────────────────
  const sMidPx = toPx(shoulderMid, w, h);
  const hMidPx = toPx(hipMid, w, h);
  const torsoVec = { x: hMidPx.x - sMidPx.x, y: hMidPx.y - sMidPx.y };
  const torsoMag = Math.sqrt(torsoVec.x ** 2 + torsoVec.y ** 2);

  // (a) Lateral lean — shoulder-hip vector vs vertical. Reliable from frontal camera.
  const lateralLeanAngle = torsoMag > 1
    ? (Math.acos(Math.max(-1, Math.min(1, torsoVec.y / torsoMag))) * 180) / Math.PI
    : 0;

  // (b) Forward lean — MediaPipe z: negative z = closer to camera = leaning forward.
  // Normalised by an approximate torso length to make it scale-independent.
  const forwardLeanProxy = Math.max(0, -shoulderMid.z * 80);

  // Combined signal: take the larger of the two
  const torsoLeanAngle = Math.max(lateralLeanAngle, forwardLeanProxy);

  // ── 4. Bharatanatyam knee tracking ───────────────────────────────────────
  // Rule: each knee must be LATERAL to its hip (wider than hip width).
  // See JSDoc and [2] note at top of file for why FPPA is not used here.
  //
  // MediaPipe x increases rightward:
  //   Left correct  → leftKnee.x  < leftHip.x  → deviation = leftHip.x  − leftKnee.x  > 0
  //   Right correct → rightKnee.x > rightHip.x → deviation = rightKnee.x − rightHip.x > 0
  const leftKneeDeviation  = lHip.x  - lKnee.x;
  const rightKneeDeviation = rKnee.x - rHip.x;

  // ── 5. Symmetry and stability ─────────────────────────────────────────────
  const kneeBendDelta    = Math.abs(leftKneeAngle - rightKneeAngle);
  const hipLevelDelta    = Math.abs(lHip.y - rHip.y);
  const footMidX         = (lAnkle.x + rAnkle.x) / 2;
  const comLateralOffset = Math.abs(hipMid.x - footMidX);

  return {
    leftKneeAngle,
    rightKneeAngle,
    avgKneeAngle,
    sittingDepthRatio,
    torsoLeanAngle,
    leftKneeDeviation,
    rightKneeDeviation,
    kneeBendDelta,
    hipLevelDelta,
    comLateralOffset,
  };
}

// ─── Scoring ───────────────────────────────────────────────────────────────────

/**
 * Thresholds calibrated against real Aramandi session data (July 2026):
 *   - Full Aramandi observed: sittingDepthRatio ≈ 0.49–0.52
 *   - Good Aramandi knee angles: 89–104°
 *   - Upright torso: 4.3°; noticeable lean: 16.7°
 */
export function scoreAramandiFeatures(
  f: AramandiFeatures
): Record<string, MetricScore> {
  const fmt = (n: number, dp = 1) => n.toFixed(dp);

  return {
    // ── Sitting Depth ──────────────────────────────────────────────────────
    // Calibrated: real full Aramandi ≈ 0.49. Threshold lowered from 0.55 → 0.44.
    sittingDepth: {
      rawValue: f.sittingDepthRatio,
      displayValue: `${(f.sittingDepthRatio * 100).toFixed(0)}%`,
      label: 'Sitting Depth',
      status:
        f.sittingDepthRatio >= 0.44 ? 'good'
        : f.sittingDepthRatio >= 0.32 ? 'warn'
        : 'error',
      feedback:
        f.sittingDepthRatio >= 0.44 ? 'Good depth'
        : f.sittingDepthRatio >= 0.32 ? 'Sit slightly deeper'
        : 'Too shallow — bend your knees further',
    },

    // ── Knee Flexion ───────────────────────────────────────────────────────
    kneeAngle: {
      rawValue: f.avgKneeAngle,
      displayValue: `${fmt(f.leftKneeAngle)}° / ${fmt(f.rightKneeAngle)}°`,
      label: 'Knee Flexion (L / R)',
      status:
        f.avgKneeAngle < 120 ? 'good'
        : f.avgKneeAngle < 150 ? 'warn'
        : 'error',
      feedback:
        f.avgKneeAngle < 120 ? 'Good knee bend'
        : f.avgKneeAngle < 150 ? 'Bend your knees further'
        : 'Knees nearly straight — this is not Aramandi',
    },

    // ── Torso Lean ────────────────────────────────────────────────────────
    // Calibrated: 4.3° = upright (good), 16.7° = noticeable lean (warn boundary at ~12°).
    torsoLean: {
      rawValue: f.torsoLeanAngle,
      displayValue: `${fmt(f.torsoLeanAngle)}°`,
      label: 'Torso Lean',
      status:
        f.torsoLeanAngle < 12 ? 'good'
        : f.torsoLeanAngle < 22 ? 'warn'
        : 'error',
      feedback:
        f.torsoLeanAngle < 12 ? 'Back is upright'
        : f.torsoLeanAngle < 22 ? 'Slight lean — lift your chest'
        : 'Significant lean — keep your spine vertical',
    },

    // ── Knee Tracking (Bharatanatyam-specific) ────────────────────────────
    // Positive = knee wider than hip (correct). Negative = knee collapsing inward.
    kneeTracking: {
      rawValue: Math.min(f.leftKneeDeviation, f.rightKneeDeviation),
      displayValue: `L ${fmt(f.leftKneeDeviation, 3)} / R ${fmt(f.rightKneeDeviation, 3)}`,
      label: 'Knee Tracking',
      status:
        f.leftKneeDeviation > 0.01 && f.rightKneeDeviation > 0.01 ? 'good'
        : f.leftKneeDeviation > -0.015 && f.rightKneeDeviation > -0.015 ? 'warn'
        : 'error',
      feedback:
        f.leftKneeDeviation > 0.01 && f.rightKneeDeviation > 0.01
          ? 'Knees tracking outward correctly'
          : f.leftKneeDeviation <= -0.015 && f.rightKneeDeviation <= -0.015
          ? 'Both knees collapsing inward — push them out'
          : f.leftKneeDeviation <= -0.015
          ? 'Left knee collapsing inward — push it out'
          : f.rightKneeDeviation <= -0.015
          ? 'Right knee collapsing inward — push it out'
          : 'Knees slightly narrow — open them wider',
    },

    // ── Knee Symmetry ─────────────────────────────────────────────────────
    symmetry: {
      rawValue: f.kneeBendDelta,
      displayValue: `${fmt(f.kneeBendDelta)}°`,
      label: 'Knee Symmetry',
      status:
        f.kneeBendDelta < 10 ? 'good'
        : f.kneeBendDelta < 18 ? 'warn'
        : 'error',
      feedback:
        f.kneeBendDelta < 10 ? 'Good left–right balance'
        : f.kneeBendDelta < 18 ? 'Slight imbalance — even out both sides'
        : 'Significant imbalance — one knee is bending much less than the other',
    },

    // ── Hip Level ─────────────────────────────────────────────────────────
    hipLevel: {
      rawValue: f.hipLevelDelta,
      displayValue: `${fmt(f.hipLevelDelta * 1000, 0)} mu`,
      label: 'Hip Level',
      status:
        f.hipLevelDelta < 0.025 ? 'good'
        : f.hipLevelDelta < 0.045 ? 'warn'
        : 'error',
      feedback:
        f.hipLevelDelta < 0.025 ? 'Hips are level'
        : f.hipLevelDelta < 0.045 ? 'Slight hip tilt — distribute weight evenly'
        : 'Hips tilting — shift weight to balance both sides',
    },

    // ── Balance / CoM ──────────────────────────────────────────────────────
    stability: {
      rawValue: f.comLateralOffset,
      displayValue: `${fmt(f.comLateralOffset * 1000, 0)} mu`,
      label: 'Balance (CoM)',
      status:
        f.comLateralOffset < 0.04 ? 'good'
        : f.comLateralOffset < 0.07 ? 'warn'
        : 'error',
      feedback:
        f.comLateralOffset < 0.04 ? 'Centre of mass is stable'
        : f.comLateralOffset < 0.07 ? 'Slight weight shift — stay centred'
        : 'Weight shifting sideways — centre yourself between both feet',
    },
  };
}