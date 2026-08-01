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

  const { reasoning, mlResult, conflict } = req;
  const top   = reasoning.topError;
  const score = reasoning.overallScore;

  const systemPrompt = `You are an experienced Bharatanatyam teacher who has been watching a student practice Aramandi. 
You have just received a biomechanical analysis of their form. 
Speak directly, warmly but honestly. Be specific — never generic. 
Do not mention AI, software, sensors, or technology. 
Respond as if you are speaking to the student face to face.`;

  const userPrompt = `
Student session analysis:
- Overall score: ${score}/100
- Session trend: ${reasoning.sessionTrend}
- Session duration: ${Math.round(req.sessionDuration / 60)} minutes
- Primary error detected: ${top ? `${top.errorKey} (${top.tier} priority)` : 'None — form looks correct'}
- Root cause identified: ${top?.rootCause ?? 'N/A'}
- This error is ${top?.isRecurring ? 'RECURRING throughout the session' : 'appearing occasionally'}
- All errors found: ${reasoning.allErrors.map(e => e.errorKey).join(', ') || 'None'}
- ML classifier verdict: ${mlResult.predictedClass} (${(mlResult.confidence * 100).toFixed(0)}% confidence)
- System conflict: ${conflict.type}

Please provide:
1. PRIMARY CORRECTION: The single most important thing they must fix (1-2 sentences, very specific)
2. WHY IT MATTERS: The pedagogical reason — what breaks in the technique if this isn't fixed (1 sentence)
3. WHAT TO DO NEXT: A specific drill or instruction for their next practice (1-2 sentences)
4. SESSION SUMMARY: A brief honest overall assessment of this session (1-2 sentences)

Format your response as JSON with keys: primaryCorrection, whyItMatters, whatToDoNext, sessionSummary
Return ONLY valid JSON, no markdown, no preamble.
  `.trim();

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model:      'claude-sonnet-4-6',
        max_tokens: 400,
        system:     systemPrompt,
        messages:   [{ role: 'user', content: userPrompt }],
      }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data  = await response.json();
    const text  = data.content[0].text as string;
    const clean = text.replace(/```json|```/g, '').trim();
    return JSON.parse(clean) as TeacherFeedback;

  } catch (err) {
    console.error('[AngaShuddhi] Teacher feedback API failed:', err);

    // Graceful fallback — static feedback if API unavailable
    return {
      primaryCorrection: top?.staticPhrase ?? 'Focus on keeping your knees outward and your back upright.',
      whyItMatters:      top?.reason ?? 'These are the foundational elements of correct Aramandi.',
      whatToDoNext:      'Practice slow Aramandi holds — lower over 4 counts, hold for 4. Quality over depth.',
      sessionSummary:    `Score: ${score}/100. ${reasoning.sessionTrend === 'improving' ? 'You improved during this session.' : reasoning.sessionTrend === 'deteriorating' ? 'Your form declined — consider rest.' : 'Consistent performance throughout.'}`,
    };
  }
}
