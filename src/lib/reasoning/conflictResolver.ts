/**
 * AngaShuddhi — Conflict Resolver
 *
 * Three-rule arbitration between rule engine and ML classifier.
 * Kept intentionally simple — the rules encode clear pedagogical logic,
 * not a complex ensemble. Complexity here adds no real benefit for MVP.
 */

import type { MetricScore }      from '../features';
import type { ClassifierResult } from '../classifier';
import type { ReasoningResult }  from './priorityEngine';

export type ConflictType =
  | 'full-agreement'      // both say error or both say correct
  | 'rules-only'          // rules flag error, ML says correct → trust rules
  | 'ml-only'             // ML flags error, rules say correct → trust ML if confident
  | 'type-disagreement';  // both flag error but disagree on type

export interface ConflictResolution {
  type:          ConflictType;
  trusted:       'rules' | 'ml' | 'both';
  explanation:   string;
  displayNote:   string | null;    // shown in UI if conflict is interesting
}

export function resolveConflict(
  ruleScores: Record<string, MetricScore>,
  mlResult:   ClassifierResult,
): ConflictResolution {

  const ruleHasError = Object.values(ruleScores).some(s => s.status !== 'good');
  const mlHasError   = mlResult.predictedClass !== 'correct';
  const mlConfident  = mlResult.confidence > 0.75;

  // ── Rule 1: Both agree there is no error ─────────────────────────────────
  if (!ruleHasError && !mlHasError) {
    return {
      type:        'full-agreement',
      trusted:     'both',
      explanation: 'Rule engine and classifier agree: pose is correct.',
      displayNote: null,
    };
  }

  // ── Rule 2: Both flag errors ──────────────────────────────────────────────
  if (ruleHasError && mlHasError) {
    return {
      type:        'full-agreement',
      trusted:     'both',
      explanation: 'Both systems detected errors. High confidence in assessment.',
      displayNote: null,
    };
  }

  // ── Rule 3: Rules flag error, ML says correct ─────────────────────────────
  // Trust rules — specific metric violations are hard evidence.
  // ML may have learned to tolerate borderline values in training data.
  if (ruleHasError && !mlHasError) {
    return {
      type:        'rules-only',
      trusted:     'rules',
      explanation: 'Specific metric thresholds violated but classifier rated pose as borderline correct. Specific measurements take precedence for per-metric feedback.',
      displayNote: null,    // don't surface this to user — rules just quietly win
    };
  }

  // ── Rule 4: ML flags error, rules say correct ─────────────────────────────
  // Trust ML only if confidence is high — it may see a multi-feature pattern
  // that no single threshold captures. If confidence is low, ignore it.
  if (!ruleHasError && mlHasError) {
    if (mlConfident) {
      return {
        type:        'ml-only',
        trusted:     'ml',
        explanation: `All individual metrics are within thresholds, but the classifier detects a holistic pattern consistent with ${mlResult.predictedClass} (${(mlResult.confidence * 100).toFixed(0)}% confidence). This suggests a subtle form issue not captured by any single metric.`,
        displayNote: `Subtle pattern detected — ${mlResult.predictedClass.replace(/_/g, ' ')}`,
      };
    } else {
      // Low confidence ML claim, no rule violation — treat as correct
      return {
        type:        'full-agreement',
        trusted:     'both',
        explanation: 'ML classification has low confidence and rules show no violations. Treating as correct.',
        displayNote: null,
      };
    }
  }

  // Fallback
  return { type: 'full-agreement', trusted: 'both', explanation: '', displayNote: null };
}


// ─────────────────────────────────────────────────────────────────────────────
// On-demand Teacher Feedback (Claude API)
// Called ONCE when user clicks "Get teacher feedback" — NOT per frame.
// Per-frame feedback uses static phrases from priorityEngine.ts.
// ─────────────────────────────────────────────────────────────────────────────

export interface TeacherFeedbackRequest {
  reasoning:        ReasoningResult;
  mlResult:         ClassifierResult;
  conflict:         ConflictResolution;
  sessionDuration:  number;    // seconds
  frameCount:       number;
}

export interface TeacherFeedback {
  primaryCorrection: string;   // the main thing to fix
  whyItMatters:      string;   // pedagogical reason
  whatToDoNext:      string;   // specific drill or instruction
  sessionSummary:    string;   // overall session assessment
}

export async function getTeacherFeedback(
  req: TeacherFeedbackRequest,
): Promise<TeacherFeedback> {
  const { reasoning } = req;
  const top   = reasoning.topError;
  const score = reasoning.overallScore;

  // Generate structured feedback from priority engine output
  // No API call needed — pedagogical intelligence is already in the reasoning layer
  return {
    primaryCorrection: top?.staticPhrase
      ?? 'Focus on keeping your knees outward and your back upright.',
    whyItMatters: top?.reason
      ?? 'These are the foundational elements of correct Aramandi.',
    whatToDoNext: getNextStep(top?.errorKey),
    sessionSummary: buildSummary(score, reasoning.sessionTrend, reasoning.allErrors.length),
  };
}

function getNextStep(errorKey: string | undefined): string {
  const drills: Record<string, string> = {
    kneeTracking:  'Practice slow Aramandi holds. Lower over 4 counts, hold for 4. If knees collapse at any point, come back up and restart. Quality over depth.',
    torsoLean:     'Try wall-back Aramandi. Stand with your back touching the wall and lower into position without your shoulders leaving the wall.',
    stability:     'Practice standing on one leg for 30 seconds each side to build balance, then return to Aramandi.',
    sittingDepth:  'Hold a doorframe for support and lower to your maximum depth. Hold for 30 seconds. Gradually increase depth each session.',
    hipLevel:      'Slow down and consciously level your hips as you descend. Use a mirror if available.',
    symmetry:      'Isolate your weaker side. Practice single-leg Aramandi preparation to build even strength.',
  };
  return drills[errorKey ?? '']
    ?? 'Practice slow Aramandi holds: 10 repetitions, 10 seconds each, focusing on the correction above.';
}

function buildSummary(
  score: number,
  trend: string,
  errorCount: number,
): string {
  const scoreText = score >= 80 ? 'Strong session'
    : score >= 60 ? 'Decent session with room to improve'
    : 'Session needs focused work';

  const trendText = trend === 'improving'    ? 'Your form improved as the session progressed.'
    : trend === 'deteriorating' ? 'Form declined toward the end — consider rest or shorter sessions.'
    : 'Consistent performance throughout.';

  const errorText = errorCount === 0
    ? 'No significant errors detected.'
    : `${errorCount} error${errorCount > 1 ? 's' : ''} detected — focus on the primary correction first.`;

  return `${scoreText}. ${trendText} ${errorText} Score: ${score}/100.`;
}