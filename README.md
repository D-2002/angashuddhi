# AngaShuddhi (अंगशुद्धि)

**Bringing computational precision to classical dance training.**

> *Anga* (body) + *Shuddhi* (purity, correctness) — the principle of precise body alignment in Bharatanatyam.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-angashuddhi.vercel.app-indigo?style=flat-square)](https://angashuddhi.vercel.app)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?style=flat-square)](https://nextjs.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-blue?style=flat-square)](https://ai.google.dev/edge/mediapipe)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## What is AngaShuddhi?

AngaShuddhi is an AI-powered movement analysis platform for Bharatanatyam students who practice without constant teacher supervision. It encodes traditional Bharatanatyam pedagogical knowledge into measurable computational metrics and delivers real-time, teacher-like feedback on foundational technique.

This is **not** a choreography generator or a generic "rate my dance" app. It is a computer vision system that quantifies foundational Bharatanatyam technique using biomechanical metrics and domain-specific pedagogical rules.

**Live at:** https://angashuddhi.vercel.app

---

## The Problem

Many Bharatanatyam students practice at home without constant teacher supervision. Foundational mistakes — collapsing knees, forward lean, insufficient sitting depth — become habits because students cannot accurately self-assess. Traditional teachers identify these issues immediately. AngaShuddhi attempts to encode that expertise computationally.

---

## Technical Architecture

```
Video Input (webcam / upload)
        ↓
MediaPipe Pose Landmarker (WASM, in-browser)
        ↓
Feature Extraction Layer
Joint angles · sitting depth · knee deviation · torso lean · CoM offset
        ↓
        ├──────────────────────┬──────────────────────┐
        │                      │                      │
  Rule Engine           ML Classifier         Pedagogical
  Hard thresholds       Random Forest         Reasoning Layer
  per metric            (ONNX, in-browser)    Priority Engine
  7 metrics             7 error classes       Causal Attribution
  independently         97.5% accuracy        Conflict Resolution
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               ↓
                    Merged Teacher Feedback
                    Score · Primary Correction · Root Cause
                               ↓
                    Supabase (session history)
```

All pose estimation and ML inference runs **client-side in the browser** via WebAssembly. No video is ever uploaded to a server. Privacy by design.

---

## The Neurosymbolic Approach

AngaShuddhi is an instance of neurosymbolic AI — a system that combines symbolic rules with neural learning, where each component addresses the other's fundamental limitations.

| Component | Strength | Limitation |
|---|---|---|
| Rule Engine | Interpretable, guaranteed, pedagogically grounded | Brittle across body proportions, no cross-metric reasoning |
| ML Classifier | Learns holistic multi-feature patterns, generalises | Black box, no causal explanation, requires labeled data |
| Reasoning Layer | Root cause attribution, pedagogical prioritisation | Relies on quality of both upstream systems |

**Key insight:** The systems are complementary, not redundant. When they disagree, the conflict is itself informative — the conflict resolver encodes the pedagogical logic for arbitration.

### Why existing biomechanics literature doesn't transfer

The standard metric for knee tracking in sports biomechanics is the **Frontal Plane Projection Angle (FPPA)** (Sigward & Powers, 2006), which compares knee position against the hip-to-ankle midline. This metric assumes parallel foot stance.

In Bharatanatyam, feet are externally rotated ~45° per side in Aramandi. This makes the ankle laterally displaced **by design**, causing FPPA to flag correct Aramandi form as valgus collapse. AngaShuddhi uses a domain-specific metric instead: **knee must be lateral to hip**, directly encoding the teacher rule *"open your knees outward."*

This is a concrete example of why domain-specific knowledge is necessary — you cannot transfer existing computer vision work to Bharatanatyam without reformulating its assumptions.

---

## Biomechanical Features

All features are computed from MediaPipe's 33-point pose landmarks. Feature engineering is grounded in published literature:

| Feature | Derivation | Reference |
|---|---|---|
| Knee flexion angle | Hip–knee–ankle angle in pixel space | Escamilla et al. (2001) |
| Sitting depth ratio | Hip descent / total body height | Bharatanatyam pedagogical rules |
| Torso lean | Shoulder–hip vector vs. vertical + MediaPipe Z depth | Standard clinical biomechanics |
| Knee tracking | Knee lateral deviation from hip (Bharatanatyam-specific) | Domain knowledge — replaces FPPA |
| CoM lateral offset | Hip midpoint vs. foot midpoint | De Leva (1996) |

---

## ML Classifier

- **Architecture:** Random Forest (200 trees, class-weighted)
- **Training data:** 8,582 labeled frames across 7 error classes, extracted from ~21 clips using MediaPipe Tasks Vision Python API
- **Features:** 10 engineered biomechanical features (not raw pixels)
- **Accuracy:** 97.5% on held-out test set (single-dancer calibration)
- **Export:** ONNX with `zipmap=False` for onnxruntime-web compatibility
- **Inference:** Fully client-side via onnxruntime-web WASM

**Honest limitation:** This accuracy reflects held-out frames from the same dancer used for training. Cross-person generalization accuracy will be lower. Expanding the training set to multiple dancers is the primary open validation task.

**Error classes:**
- `correct`
- `knee_collapse` (bilateral)
- `left_knee_collapse`
- `right_knee_collapse`
- `forward_lean`
- `insufficient_depth`
- `weight_shift`

---

## Pedagogical Reasoning Layer

The reasoning layer is the primary intellectual contribution. It transforms raw system outputs into teacher-like assessment through three modules:

### Priority Engine
Encodes Bharatanatyam pedagogical hierarchy — teachers correct the most fundamental error first, not all errors simultaneously.

```
Fatal (priority 90–100):   knee collapse · forward lean · weight shift
Structural (60–75):        sitting depth · foot spacing
Refinement (35–45):        symmetry · hip level
```

Recurring errors (appearing in >60% of frames) receive a priority boost — consistent errors indicate a deeper technique gap than occasional ones.

### Causal Fault Tree
Identifies upstream root causes rather than labelling observed symptoms. Examples:

- Bilateral knee collapse + shallow depth → likely root cause: **insufficient foot spacing**
- Weight shift + forward lean → likely root cause: **forward lean displacing centre of mass**
- Single-sided knee collapse → likely root cause: **limited hip external rotation on that side**

This mirrors how an experienced teacher thinks: fix the root, not the symptom.

### Conflict Resolver
Three-rule arbitration when rule engine and classifier disagree:
1. Rule flags error, ML says correct → trust rules (specific measurements are hard evidence)
2. ML flags error with >75% confidence, rules say correct → surface ML pattern (holistic pattern not captured by single metric)
3. Both flag errors → high confidence, surface both

---

## Supported Poses (V1)

**Static:**
- Aramandi (half-sitting position)

**Planned (V2):**
- Muzhumandi
- Tatta Adavu
- Natta Adavu

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, TypeScript, Tailwind CSS |
| Pose Estimation | MediaPipe Tasks Vision (WASM, in-browser) |
| ML Inference | onnxruntime-web (WASM, in-browser) |
| ML Training | scikit-learn, skl2onnx (Google Colab) |
| Feature Extraction | Custom TypeScript (mirrors Python training exactly) |
| LLM Feedback | Groq (llama-3.1-8b-instant) via Next.js API route |
| Analytics | Vercel Analytics |
| Deployment | Vercel |

---

## Local Development

```bash
git clone https://github.com/D-2002/angashuddhi
cd angashuddhi
npm install
```

Create `.env.local`:
```
GROQ_API_KEY=your_groq_key_here
```

```bash
npm run dev
# Open http://localhost:3000
```

**Model files** (`public/model/`) are committed to the repo:
- `angashuddhi_model.onnx` — trained Random Forest classifier
- `angashuddhi_scaler.json` — feature scaler parameters

---

## Dataset

The training dataset was created specifically for this project — no labeled Bharatanatyam pose dataset exists publicly.

**Collection:** ~21 short video clips filmed by the project author (9 years Bharatanatyam training, national-level certification), covering 7 error classes with deliberate technique variations.

**Extraction:** MediaPipe Tasks Vision Python API, sampling every 3rd frame (~10fps), with visibility filtering (min landmark visibility > 0.3) to exclude occluded frames.

**Augmentation:** Frame-level sampling from video provides natural pose variation within each class. Future work: biomechanically-constrained synthetic augmentation of landmark coordinates.

---

## Research Context and Open Questions

AngaShuddhi was built as a portfolio project for MS Computer Science applications, motivated by a genuine research question:

> *How should symbolic domain knowledge be integrated with data-driven classifiers for physical skill assessment, when training data is scarce, interpretability is required, and the domain actively violates assumptions in existing computer vision literature?*

**Open questions this project surfaces:**
- Does the system generalize across body proportions? (untested — current calibration is single-dancer)
- Can symbolic constraints act as regularization when labeled data is sparse?
- How should conflicting outputs from rule and neural systems be arbitrated in general?
- Can pedagogical rules be learned from expert knowledge rather than hand-coded — and what is lost when they are?
- Would a side-view camera or depth sensor meaningfully improve torso lean detection?

These questions point toward neurosymbolic AI, knowledge-infused learning, and multi-view pose estimation as natural research directions.

---

## Why Bharatanatyam is Computationally Interesting

Beyond personal motivation (9 years of training), Bharatanatyam is a particularly challenging domain for computer vision:

1. **Turned-out foot stance breaks existing metrics.** Standard biomechanics metrics assume parallel feet. Bharatanatyam's mandatory ~45° external rotation per foot makes standard knee tracking metrics produce inverted results.

2. **Simultaneous multi-plane constraints.** Aramandi requires simultaneous satisfaction of constraints across frontal, sagittal, and transverse planes — standard per-joint evaluation misses the causal structure.

3. **Subtle discriminative signals.** Errors that are obvious to a trained teacher are visually indistinguishable to a casual observer — the discriminative information is in biomechanical features, not raw visual appearance.

4. **Codified geometry.** Classical Bharatanatyam positions are specified in the Natyashastra (2000-year-old Sanskrit treatise) — the geometry IS the cultural transmission mechanism. This creates a rare alignment between symbolic rules and cultural knowledge.

5. **Guaranteed data scarcity.** No labeled Bharatanatyam pose dataset exists, making it a natural testbed for few-shot and knowledge-infused learning approaches.

---

## References

- Escamilla et al. (2001). Knee biomechanics of the dynamic squat exercise. *Medicine & Science in Sports & Exercise*, 33(1), 127–141.
- Sigward & Powers (2006). The influence of gender on knee kinematics, kinetics and muscle activation patterns. *Clinical Biomechanics*, 21(1), 41–48. *(Note: FPPA from this paper is intentionally not used — see architecture section)*
- Coplan (2002). Ballet dancer's turnout and its relationship to self-reported injury. *JOSPT*, 32(11), 579–584.
- De Leva (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223–1230.

---

## Author

Built by Diya Ramani — SDE at Barclays, B.Tech IT from VIT Vellore (CGPA 9.39), 9 years Bharatanatyam training.

*This project is part of a portfolio for MS Computer Science applications, with research interests in neurosymbolic AI, knowledge-based learning, and representational learning.*