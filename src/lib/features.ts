/**
 * AngaShuddhi — Feature Extraction Layer (Day 2)
 *
 * Converts raw MediaPipe Pose landmarks into Bharatanatyam-relevant
 * biomechanical metrics for Aramandi assessment.
 *
 * Coordinate system (MediaPipe normalized):
 *   x ∈ [0,1]  — left → right edge of image
 *   y ∈ [0,1]  — top → bottom (y increases downward)
 *   z           — depth relative to hip midpoint, same scale as x
 *
 * Key references:
 * [1] Escamilla et al. (2001). Knee biomechanics of the dynamic squat exercise.
 *     Medicine & Science in Sports & Exercise, 33(1), 127–141.
 * [2] Sigward & Powers (2006). The influence of gender on knee kinematics,
 *     kinetics and muscle activation patterns during side-step cutting.
 *     Clinical Biomechanics, 21(1), 41–48.
 * [3] Coplan (2002). Ballet dancer's turnout and its relationship to
 *     self-reported injury. JOSPT, 32(11), 579–584.
 * [4] De Leva (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia
 *     parameters. Journal of Biomechanics, 29(9), 1223–1230.
 */

// ─── Types ────────────────────────────────────────────────────────────────────

/** Matches MediaPipe's NormalizedLandmark shape — no extra import needed. */
export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface AramandiFeatures {
  /**
   * Hip–Knee–Ankle angle in degrees (pixel-space, aspect-ratio corrected).
   * 180° = straight leg.  ~90–110° = good Aramandi depth.
   * Ref: Escamilla 2001 — deep squat target range.
   */
  leftKneeAngle: number;
  rightKneeAngle: number;
  avgKneeAngle: number;

  /**
   * How deeply the dancer is sitting: 0 = standing, 1 = hips at ankle level.
   * Derived from hip midpoint y-position relative to full standing height.
   * Target for Aramandi: > 0.55
   */
  sittingDepthRatio: number;

  /**
   * Trunk lean from vertical in degrees.
   * Vector: shoulder midpoint → hip midpoint vs. vertical axis.
   * Target: < 10° (upright back).
   */
  torsoLeanAngle: number;

  /**
   * Signed lateral deviation of each knee from the hip–ankle midline (norm. units).
   * Positive = knee tracking outward (correct Aramandi turnout).
   * Negative = knee caving inward (valgus collapse — most common error).
   * Ref: Sigward & Powers 2006 (FPPA).
   */
  leftKneeDeviation: number;
  rightKneeDeviation: number;

  /**
   * |leftKneeAngle − rightKneeAngle|.
   * Measures left–right bend symmetry.  Target: < 10°.
   */
  kneeBendDelta: number;

  /**
   * |leftHip.y − rightHip.y| in normalized units.
   * Measures hip tilt / weight shift.  Target: < 0.025.
   */
  hipLevelDelta: number;

  /**
   * |hipMidpoint.x − footMidpoint.x| in normalized units.
   * Approximate lateral CoM offset.  Ref: De Leva 1996.
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

// ─── Geometry helpers ─────────────────────────────────────────────────────────

type P2 = { x: number; y: number };

/**
 * Angle at vertex B formed by rays B→A and B→C, in degrees.
 * Works in any 2D coordinate space — pass pixel-space points for
 * aspect-ratio-correct results.
 */
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

/** Convert normalized landmark → pixel-space point. */
function toPx(lm: Landmark, w: number, h: number): P2 {
  return { x: lm.x * w, y: lm.y * h };
}

/** Midpoint between two landmarks. */
function midLm(a: Landmark, b: Landmark): Landmark {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
  };
}

// ─── Feature extraction ───────────────────────────────────────────────────────

/**
 * Extract all Aramandi features from one frame of MediaPipe landmarks.
 *
 * @param lm  33-element array from PoseLandmarker results.landmarks[i]
 * @param w   Frame width in pixels  (canvas.width)
 * @param h   Frame height in pixels (canvas.height)
 */
export function extractAramandiFeatures(
  lm: Landmark[],
  w: number,
  h: number
): AramandiFeatures {
  // ── Named references (MediaPipe 33-point body model) ──
  const lShoulder = lm[11], rShoulder = lm[12];
  const lHip      = lm[23], rHip      = lm[24];
  const lKnee     = lm[25], rKnee     = lm[26];
  const lAnkle    = lm[27], rAnkle    = lm[28];

  // ── 1. Knee flexion angles ────────────────────────────────────────────────
  // Convert to pixel space first to correct for aspect ratio distortion.
  const leftKneeAngle  = angleDeg(toPx(lHip, w, h),  toPx(lKnee, w, h),  toPx(lAnkle, w, h));
  const rightKneeAngle = angleDeg(toPx(rHip, w, h),  toPx(rKnee, w, h),  toPx(rAnkle, w, h));
  const avgKneeAngle   = (leftKneeAngle + rightKneeAngle) / 2;

  // ── 2. Sitting depth ratio ────────────────────────────────────────────────
  // Pure vertical comparison — normalized y works fine (no aspect ratio issue).
  const hipMid      = midLm(lHip, rHip);
  const shoulderMid = midLm(lShoulder, rShoulder);
  const ankleMidY   = (lAnkle.y + rAnkle.y) / 2;
  const totalHeight = ankleMidY - shoulderMid.y;   // full height reference
  const hipDescent  = hipMid.y - shoulderMid.y;    // hip's position within that range
  // 0 = hips at shoulder level (standing), 1 = hips at ankle level (max squat)
  const sittingDepthRatio = totalHeight > 0.05
    ? Math.min(1, Math.max(0, hipDescent / totalHeight))
    : 0;

  // ── 3. Torso lean from vertical ───────────────────────────────────────────
  const sMidPx = toPx(shoulderMid, w, h);
  const hMidPx = toPx(hipMid, w, h);
  const torsoVec = { x: hMidPx.x - sMidPx.x, y: hMidPx.y - sMidPx.y };
  const torsoMag = Math.sqrt(torsoVec.x ** 2 + torsoVec.y ** 2);
  // Dot product with (0,1) = vertical downward direction
  const torsoLeanAngle = torsoMag > 1
    ? (Math.acos(Math.max(-1, Math.min(1, torsoVec.y / torsoMag))) * 180) / Math.PI
    : 0;

  // ── 4. Knee lateral deviation (FPPA proxy) ────────────────────────────────
  // Each knee is compared to the midpoint of its hip and ankle x-positions.
  // Positive = knee lateral to that midline (correct outward tracking).
  // Negative = knee medial to midline (valgus collapse).
  const lMidX = (lHip.x + lAnkle.x) / 2;
  const rMidX = (rHip.x + rAnkle.x) / 2;
  const leftKneeDeviation  = lMidX - lKnee.x;  // left leg: outward = smaller x
  const rightKneeDeviation = rKnee.x - rMidX;  // right leg: outward = larger x

  // ── 5. Symmetry & stability ───────────────────────────────────────────────
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

// ─── Scoring ──────────────────────────────────────────────────────────────────

/**
 * Convert raw features into colour-coded, feedback-carrying metric scores.
 *
 * Thresholds grounded in:
 *   - Knee angle:    Escamilla 2001 (90–110° for deep squat)
 *   - Torso lean:    standard clinical biomechanics (<10°)
 *   - FPPA:          Sigward & Powers 2006
 *   - Sitting depth: Bharatanatyam pedagogical rules (domain knowledge)
 */
export function scoreAramandiFeatures(
  f: AramandiFeatures
): Record<string, MetricScore> {
  const fmt = (n: number, dp = 1) => n.toFixed(dp);

  return {
    sittingDepth: {
      rawValue: f.sittingDepthRatio,
      displayValue: `${(f.sittingDepthRatio * 100).toFixed(0)}%`,
      label: 'Sitting Depth',
      status:
        f.sittingDepthRatio >= 0.55 ? 'good'
        : f.sittingDepthRatio >= 0.40 ? 'warn'
        : 'error',
      feedback:
        f.sittingDepthRatio >= 0.55 ? 'Good depth — hold it'
        : f.sittingDepthRatio >= 0.40 ? 'Sit slightly deeper'
        : 'Insufficient depth — bend your knees more',
    },

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
        : f.avgKneeAngle < 150 ? 'Bend your knees further into the position'
        : 'Knees are nearly straight — you are not in Aramandi',
    },

    torsoLean: {
      rawValue: f.torsoLeanAngle,
      displayValue: `${fmt(f.torsoLeanAngle)}°`,
      label: 'Torso Lean',
      status:
        f.torsoLeanAngle < 10 ? 'good'
        : f.torsoLeanAngle < 22 ? 'warn'
        : 'error',
      feedback:
        f.torsoLeanAngle < 10 ? 'Back is upright'
        : f.torsoLeanAngle < 22 ? 'Slight forward lean — lift your chest'
        : 'You are leaning forward — keep your spine vertical',
    },

    kneeTracking: {
      rawValue: Math.min(f.leftKneeDeviation, f.rightKneeDeviation),
      displayValue: `L ${fmt(f.leftKneeDeviation, 3)} / R ${fmt(f.rightKneeDeviation, 3)}`,
      label: 'Knee Tracking',
      status:
        f.leftKneeDeviation > 0 && f.rightKneeDeviation > 0 ? 'good'
        : f.leftKneeDeviation > -0.02 && f.rightKneeDeviation > -0.02 ? 'warn'
        : 'error',
      feedback:
        f.leftKneeDeviation > 0 && f.rightKneeDeviation > 0
          ? 'Knees tracking outward correctly'
          : f.leftKneeDeviation <= 0 && f.rightKneeDeviation <= 0
          ? 'Both knees are collapsing inward — push them out'
          : f.leftKneeDeviation <= 0
          ? 'Left knee is collapsing inward — push it out'
          : 'Right knee is collapsing inward — push it out',
    },

    symmetry: {
      rawValue: f.kneeBendDelta,
      displayValue: `${fmt(f.kneeBendDelta)}°`,
      label: 'Knee Symmetry',
      status:
        f.kneeBendDelta < 8 ? 'good'
        : f.kneeBendDelta < 16 ? 'warn'
        : 'error',
      feedback:
        f.kneeBendDelta < 8 ? 'Good left–right balance'
        : f.kneeBendDelta < 16 ? 'Slight imbalance — even out both sides'
        : 'Significant imbalance — one knee is bending much less than the other',
    },

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
        : 'Hips are tilting — shift weight to balance both sides',
    },

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
        : 'Weight is shifting sideways — centre yourself between both feet',
    },
  };
}