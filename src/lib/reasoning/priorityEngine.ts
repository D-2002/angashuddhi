/**
 * AngaShuddhi — Priority Engine
 *
 * The single most teacher-like behaviour: given multiple errors,
 * identify the one that matters most RIGHT NOW and explain why.
 *
 * Pedagogical principle: a student given five corrections simultaneously
 * fixes zero of them. Surface one correction, ordered by:
 *   1. Fatal errors (knee collapse, forward lean, weight shift)
 *   2. Structural errors (depth, foot spacing)
 *   3. Refinement errors (symmetry, hip level)
 *
 * The causal relationships encoded as SPECIAL_CASES below are a simplified
 * form of the Pedagogical Causal Graph — MVP version that preserves the
 * key insight (correct root causes, not symptoms) without full graph traversal.
 */

import type { MetricScore, AramandiFeatures } from '../features';
import type { ClassifierResult } from '../classifier';

// ─── Types ────────────────────────────────────────────────────────────────────

export type ErrorTier = 'fatal' | 'structural' | 'refinement' | 'none';

export interface PriorityError {
  tier:         ErrorTier;
  errorKey:     string;           // matches rule engine key
  priority:     number;           // higher = more urgent (0–100)
  headline:     string;           // one-line correction
  reason:       string;           // why this is the priority
  rootCause:    string;           // what upstream issue likely caused this
  staticPhrase: string;           // immediate teacher-voice feedback for per-frame display
  isRecurring:  boolean;
}

export interface ReasoningResult {
  topError:        PriorityError | null;
  allErrors:       PriorityError[];
  overallScore:    number;         // 0–100
  sessionTrend:    'improving' | 'stable' | 'deteriorating';
  phraseToDisplay: string;        // what to show in the UI right now
}

// ─── Pedagogical hierarchy ─────────────────────────────────────────────────────

// Priority values — never reorder without a pedagogical reason
const PRIORITY_MAP: Record<string, { priority: number; tier: ErrorTier }> = {
  kneeTracking:  { priority: 100, tier: 'fatal'       },   // most critical — everything else depends on it
  torsoLean:     { priority:  90, tier: 'fatal'       },   // forward lean breaks posture chain
  stability:     { priority:  85, tier: 'fatal'       },   // weight shift = unstable base
  sittingDepth:  { priority:  70, tier: 'structural'  },   // depth after knees are tracking
  hipLevel:      { priority:  60, tier: 'structural'  },   // hip tilt once weight is stable
  symmetry:      { priority:  40, tier: 'refinement'  },   // fine-tuning once structure is right
  kneeAngle:     { priority:  35, tier: 'refinement'  },
};

// ─── Causal special cases (simplified causal graph) ──────────────────────────
// If knee_collapse observed, check if foot spacing might be upstream cause.
// We infer this from feature values rather than a traversable graph.
function inferRootCause(
  errorKey: string,
  features: AramandiFeatures,
): { rootCause: string; reason: string } {

  if (errorKey === 'kneeTracking') {
    // Heuristic: if both knees are collapsing AND sitting depth is shallow,
    // the upstream cause is likely insufficient foot spacing — a narrow stance
    // prevents correct knee tracking regardless of intent.
    const bothCollapsing = features.leftKneeDeviation < 0 && features.rightKneeDeviation < 0;
    if (bothCollapsing && features.sittingDepthRatio < 0.4) {
      return {
        rootCause: 'Narrow foot stance',
        reason: 'When both knees collapse simultaneously with shallow depth, the root cause is usually foot placement that is too narrow. The geometry of the hip-knee-ankle chain does not allow correct knee tracking unless the feet are wide enough.',
      };
    }
    // If only one knee collapses, it is more likely a hip rotation issue on that side
    if (features.leftKneeDeviation < -0.05 && features.rightKneeDeviation > 0) {
      return {
        rootCause: 'Left hip external rotation',
        reason: 'Single-sided knee collapse usually indicates limited hip external rotation on that side rather than a foot spacing problem.',
      };
    }
    if (features.rightKneeDeviation < -0.05 && features.leftKneeDeviation > 0) {
      return {
        rootCause: 'Right hip external rotation',
        reason: 'Single-sided knee collapse usually indicates limited hip external rotation on that side rather than a foot spacing problem.',
      };
    }
  }

  if (errorKey === 'stability') {
    // Weight shift often caused by forward lean displacing CoM
    if (features.torsoLeanAngle > 12) {
      return {
        rootCause: 'Forward lean displacing centre of mass',
        reason: 'Weight shift is often a downstream consequence of forward lean — the torso displacement shifts the centre of mass sideways, which the body compensates by shifting weight.',
      };
    }
  }

  if (errorKey === 'sittingDepth') {
    // Shallow depth often caused by knee collapse preventing deeper descent
    if (features.leftKneeDeviation < -0.04 || features.rightKneeDeviation < -0.04) {
      return {
        rootCause: 'Knee collapse limiting descent',
        reason: 'Shallow depth is often caused by collapsing knees — the body stays higher to maintain balance when the knees are not tracking outward correctly. Fix the knee tracking first and depth often improves.',
      };
    }
  }

  // Default: no upstream cause identified
  return { rootCause: 'Direct technique error', reason: '' };
}

// ─── Teacher phrases per error — varied, specific, teacher-voice ─────────────

const PHRASES_BY_ERROR: Record<string, string[]> = {
  kneeTracking: [
    'Push your knees outward — the rotation comes from the hip, not the knee.',
    'Your knees are narrowing. Widen your stance first, then rotate from the hip.',
    'Knees should be wider than your hips in Aramandi. Push them back and out.',
    'Feel the floor with your outer feet and use that pressure to drive the knees out.',
  ],
  torsoLean: [
    'Chest up — your back is drifting forward. Engage your core and lift.',
    'Imagine a wall behind you. Your spine should stay vertical.',
    'Lift your sternum and pull your shoulders back. The torso should be upright.',
    'Forward lean is your body compensating for instability. Root your feet and lift your chest.',
  ],
  stability: [
    'Weight is shifting — press both feet equally into the floor.',
    'Find your midpoint between both feet and root yourself there.',
    'Your centre is moving. Still the body first, then focus on form.',
    'Equal weight on both feet. Feel the tripod of each foot — heel, big toe, little toe.',
  ],
  sittingDepth: [
    'Sit deeper — bend the knees further and lower your hips.',
    'You are too high. Aramandi requires a deeper knee bend.',
    'Lower slowly. Take 4 counts to descend if needed, but go deeper.',
    'The depth should feel uncomfortable at first — that is where the training is.',
  ],
  hipLevel: [
    'Hips are tilting — distribute your weight evenly between both legs.',
    'One side is lower than the other. Level your hips.',
    'Your hip line should be parallel to the floor. Adjust your weight distribution.',
  ],
  symmetry: [
    'One knee is bending more than the other — even out both sides.',
    'Left and right are not matching. Focus on the weaker side.',
    'Symmetry is essential. Both knees should be at the same angle.',
  ],
};

function pickPhrase(errorKey: string, isRecurring: boolean): string {
  const phrases = PHRASES_BY_ERROR[errorKey] ?? ['Focus on your form.'];
  const base    = phrases[Math.floor(Math.random() * phrases.length)];
  if (isRecurring) {
    return `I keep seeing this — ${base.charAt(0).toLowerCase() + base.slice(1)}`;
  }
  return base;
}

// ─── Recurring error detection ────────────────────────────────────────────────

export function detectRecurringErrors(
  errorHistory: string[][],    // last N frames, each is a list of error keys
  threshold = 0.6,             // error appearing in >60% of frames = recurring
): Set<string> {
  if (errorHistory.length < 10) return new Set();

  const counts: Record<string, number> = {};
  for (const frameErrors of errorHistory) {
    for (const err of frameErrors) {
      counts[err] = (counts[err] ?? 0) + 1;
    }
  }

  const recurring = new Set<string>();
  for (const [err, count] of Object.entries(counts)) {
    if (count / errorHistory.length > threshold) {
      recurring.add(err);
    }
  }
  return recurring;
}

// ─── Session trend ────────────────────────────────────────────────────────────

export function computeSessionTrend(
  scoreHistory: number[],
): 'improving' | 'stable' | 'deteriorating' {
  if (scoreHistory.length < 20) return 'stable';

  const third  = Math.floor(scoreHistory.length / 3);
  const early  = scoreHistory.slice(0, third);
  const late   = scoreHistory.slice(-third);
  const avgEarly = early.reduce((s, v) => s + v, 0) / early.length;
  const avgLate  = late.reduce((s, v)  => s + v, 0) / late.length;

  if (avgLate < avgEarly - 6) return 'deteriorating';
  if (avgLate > avgEarly + 6) return 'improving';
  return 'stable';
}

// ─── Score computation ────────────────────────────────────────────────────────

function computeScore(errors: PriorityError[]): number {
  let score = 100;
  for (const err of errors) {
    const deduction =
      err.tier === 'fatal'       ? 25 :
      err.tier === 'structural'  ? 15 :
      err.tier === 'refinement'  ? 8  : 0;
    score -= deduction;
    if (err.isRecurring) score -= 5;
  }
  return Math.max(0, Math.min(100, score));
}

// ─── Main: run priority engine ─────────────────────────────────────────────────

export function runPriorityEngine(
  ruleScores:   Record<string, MetricScore>,
  features:     AramandiFeatures,
  errorHistory: string[][],
  scoreHistory: number[],
): ReasoningResult {

  const recurringErrors = detectRecurringErrors(errorHistory);

  // Collect all active errors from rule engine
  const activeErrors: PriorityError[] = [];

  for (const [key, score] of Object.entries(ruleScores)) {
    if (score.status === 'good') continue;

    const config      = PRIORITY_MAP[key];
    if (!config) continue;

    const isRecurring  = recurringErrors.has(key);
    const causal       = inferRootCause(key, features);
    const recPenalty   = isRecurring ? 10 : 0;    // recurring errors get priority boost

    activeErrors.push({
      tier:         config.tier,
      errorKey:     key,
      priority:     config.priority + recPenalty + (score.status === 'error' ? 5 : 0),
      headline:     score.feedback,
      reason:       causal.reason || `${score.label} is outside the acceptable range.`,
      rootCause:    causal.rootCause,
      staticPhrase: pickPhrase(key, isRecurring),
      isRecurring,
    });
  }

  // Sort by priority — highest first
  activeErrors.sort((a, b) => b.priority - a.priority);

  const topError      = activeErrors[0] ?? null;
  const overallScore  = computeScore(activeErrors);
  const sessionTrend  = computeSessionTrend(scoreHistory);

  // What to show in the UI this frame
  let phraseToDisplay = 'Hold the position. Focus on stability.';
  if (topError) {
    phraseToDisplay = topError.staticPhrase;
  }
  if (activeErrors.length === 0) {
    phraseToDisplay = sessionTrend === 'improving'
      ? 'Looking better — maintain this.'
      : 'Good form. Keep holding.';
  }

  return {
    topError,
    allErrors:    activeErrors,
    overallScore,
    sessionTrend,
    phraseToDisplay,
  };
}
