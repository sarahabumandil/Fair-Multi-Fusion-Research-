# Research Proposal

**Project:** Modality Fairness in Multimodal AI — Theoretical Framework and Audio-Visual Fusion Prototype
**Course:** CIR Course Research Project
**Industry Alignment:** Nabra Project (in collaboration with a Saudi Arabian partner company)
**Research Team:** Sarah (Team Leader), Asala, Medhat, Mohamed
**Supervision:** Course Professor

---

## 1. Suggested Research Titles

1. **"Beyond the Traditional Triad: A Unified Framework for Modality Fairness in Multimodal AI"**
2. **"Silencing the Signal: Rethinking Physiological and Environmental Data as First-Class Modalities in Multimodal Learning"**
3. **"From Noise to Signal: A Six-Modality Framework for Equitable Multimodal Fusion, with an Audio-Visual Case Study"**

---

## 2. Research Problem Statement

Modern multimodal AI systems are built on an unstated hierarchy. Across the literature, "multimodal" has come to mean, almost exclusively, some subset of the **Traditional Triad**: Audio, Vision, and Text. Surveys of the field describe multimodal fusion almost entirely through this lens — sentiment analysis, audio-visual emotion recognition, vision-language pretraining — while six other classes of signal that humans and embedded systems routinely produce and rely on (Touch & Motion, Physiological Indicators, Environmental Sensors, and others) are treated, at best, as auxiliary inputs, and at worst, as noise to be filtered out rather than information to be exploited.

This is not a cosmetic omission. The comprehensive survey on multimodal fusion under low-quality data explicitly frames its four core challenges — noisy data, incomplete data, **imbalanced data**, and quality-varying data — around the assumption that some modalities are inherently weaker or less reliable, and that the central engineering task is to prevent a "performance-dominant" modality from suppressing the others. Even the most recent and most rigorous work on modality imbalance (PDMP, ARL) operates entirely within audio-visual or audio-visual-text settings, mathematically formalizing how one modality should "dominate" another. The frameworks are technically sophisticated, but they are built, end to end, on a closed universe of two or three modalities. None of the five papers reviewed for this proposal — not the PDMP paper, not the Imbalanced Learning (ARL) paper, not either of the two emotion-recognition fusion papers, not even the comprehensive low-quality-data survey — considers physiological signals (heart rate, EDA, EEG-type indicators), biomechanical/motion data, or ambient environmental sensing as a modality class in its own right.

The practical cost of this exclusion is significant. A system that fuses only audio and vision to recognize emotion, as in the cross-attention and bi-directional attention architectures reviewed below, is structurally blind to a panic attack that produces no distinctive facial expression but a measurable spike in heart rate; it cannot use ambient temperature or noise-floor data to contextualize a flagged event; it has no path to incorporate gesture, posture, or touch dynamics except by encoding them, after the fact, as if they were video. As multimodal systems move out of curated lab benchmarks (CREMA-D, AVE, AffWild2) into real deployment — wearables, smart environments, human-robot interaction, the kind of embedded, sensor-rich settings the Nabra project targets — the cost of architecturally excluding physiological and environmental signal classes compounds. The field does not currently have a principled framework for *where these signals belong* in a fusion architecture, nor for *how fairly to weight them once they are included*. A well-defined problem is half the solution: this project's problem is that "modality imbalance" research has so far been solved only inside a truncated modality space, and the field lacks a theoretical scaffold for extending fairness guarantees to the modalities it has excluded.

---

## 3. Smart Literature Review & Synthesis

### A. What has been achieved in traditional Audio-Visual-Text fusion

The reviewed literature shows a clear maturation arc within the Traditional Triad.

At the **architecture level**, audio-visual fusion has progressed from naive concatenation toward genuinely interactive mechanisms. The Joint Cross-Attention model for audio-visual dimensional emotion recognition demonstrates that fusing a *joint* audio-visual representation back into a cross-attention computation — rather than simply letting one modality's features attend to the other's — better captures both intra- and inter-modal correlation, and outperforms vanilla cross-attention and leader-follower attention on the AffWild2 benchmark, lifting valence CCC from a 0.180 unimodal baseline to 0.374 in fusion. The bi-directional cross-attention and temporal-modeling framework for the ABAW EXPR task pushes this further by letting visual and audio streams query each other symmetrically, adding a Temporal Convolutional Network for frame-level temporal dependencies in facial sequences and a text-guided contrastive objective borrowed from CLIP, improving Macro F1 from a 0.250 baseline to 0.333.

At the **optimization level**, a separate and increasingly influential line of work has identified *why* naive joint training underperforms even when fusion architecture is sophisticated. PDMP shows that the long-standing assumption — that multimodal underperformance is caused by *imbalanced* learning between modalities, and that the fix is to *balance* gradients — is actually wrong in general: the paper proves, both theoretically (via an information-theoretic mutual-information criterion) and empirically (CREMA-D, AVE, Kinetics-Sounds, CEFA, UCF-101, VGGSound), that multimodal performance is optimized when the *performance-dominant* modality is allowed to dominate optimization, not suppressed toward equality. The companion Imbalanced Learning (ARL) paper extends this further with a bias-variance decomposition, proving that the optimal contribution ratio between modalities is the *inverse of their prediction-variance ratio*, not 1:1 — and that this ratio is not even fixed across datasets (it is 3:1 in their AVE experiments, not 1:1), which is itself evidence that "balance" is the wrong global objective for multimodal systems.

The comprehensive low-quality-data survey synthesizes this entire space into four challenge categories (noisy, incomplete, imbalanced, quality-varying multimodal data) and shows that essentially every architectural innovation in the field — gradient modulation, architecture-based rebalancing, uncertainty-aware dynamic fusion, Dempster-Shafer evidence combination — has been developed and validated almost exclusively on audio-visual or audio-visual-text combinations.

### B. What remains critically missing

Three gaps recur across all five papers, and none of them is addressed by extending audio-visual techniques marginally — they require a different starting framework.

**First, modality scope.** Every dataset cited across PDMP, ARL, and the survey (CREMA-D, AVE, Kinetics-Sounds, CEFA, UCF-101, VGGSound, MOSI, AffWild2) is built from audio, vision, optical flow, or text. CEFA's RGB/Depth/IR setup is the closest any of these papers come to a non-traditional sensing modality, and even that is still camera-derived. Physiological signals (heart rate variability, EDA, EEG), motion/biomechanical signals beyond optical flow, and environmental sensor streams (ambient temperature, light, proximity) appear nowhere as first-class fusion inputs in any of the five papers.

**Second, the imbalance theory itself is modality-count-agnostic but has never been stress-tested on heterogeneous, non-visual sensing.** PDMP's performance-dominant-modality mining and ARL's variance-ratio weighting are both *architecture-independent* — they operate purely on logit outputs and unimodal performance rankings — which means, in principle, the theory could extend to six modalities including physiological and environmental signals. But the bias-variance derivation in ARL is explicitly worked out for the two-modality case, and its extension to higher-cardinality, more heterogeneous modality sets (where "variance" might mean something different for a noisy EDA sensor than for a video frame) is left as an open question. The PDMP paper's own discussion section gestures at this limitation by noting its experiments stop at three modalities (CEFA).

**Third, the missing-modality and noise literature treats absence and corruption probabilistically, but never asks whether some modalities are *structurally* prevented from being captured in the first place.** The survey's incomplete-multimodal-learning section catalogs imputation and imputation-free strategies extensively, but the implicit assumption throughout is that all modalities were *intended* to be collected and merely failed to arrive (sensor failure, patient refusal of a PET scan). It does not address the prior-stage problem this proposal is centered on: a system architecture that never collects physiological or environmental data at all, and so has no "missing modality" to impute — the modality was never in the design space to begin with.

### C. How this work bridges the gap

This project addresses the gap at two levels, deliberately scoped to be complementary rather than redundant.

**Theoretically**, we build a six-modality unified framework (Language, Audio, Vision, Touch & Motion, Physiological Indicators, Environmental Sensors) that explicitly positions physiological and environmental signals as legitimate fusion inputs rather than noise, and examines whether the performance-dominant-modality logic from PDMP and the variance-ratio weighting from ARL generalize beyond the two- and three-modality cases tested in the literature.

**Practically**, rather than attempting to build and validate a six-sensor prototype within a 20-day window — which the reviewed literature itself suggests is methodologically risky, since modality analysis costs and architecture validation scale with each added modality — we ground the empirical component in the best-validated, most reproducible part of the literature: audio-visual fusion. The cross-attention architectures in the two emotion-recognition papers, combined with the imbalance-aware optimization strategies from PDMP and ARL, give us a working, benchmarked foundation. The prototype therefore serves two purposes: it is a legitimate audio-visual fusion system in its own right, and it is also a controlled testbed for asking whether modality-fairness techniques validated on two modalities behave the way the six-modality theoretical framework predicts they should — a first empirical bridge between the theoretical and practical halves of this proposal.

---

## 4. Formulated Research Questions & Gap

**Theoretical questions** (unanswered by the surveyed literature):

- **RQ1:** Can a unified taxonomy meaningfully classify Touch & Motion, Physiological Indicators, and Environmental Sensor data alongside Language, Audio, and Vision within the existing data-centric "low-quality multimodal data" taxonomy (noisy / incomplete / imbalanced / quality-varying), or do these signal classes require new challenge categories the survey does not anticipate?
- **RQ2:** Does the PDMP finding — that imbalanced optimization driven by the performance-dominant modality outperforms balanced optimization — hold when the "weak" modality is not simply a lower-accuracy version of the same prediction task (as audio is to vision in AVE/CREMA-D), but a structurally different signal type, such as a physiological indicator that is sparse, low-dimensional, and only intermittently informative?
- **RQ3:** Does the ARL bias-variance framework's inverse-variance weighting rule extend cleanly past two modalities, or does the "numerical solution is meaningless" failure mode the ARL paper identifies for the bias term (Equations 11–12 in ARL) become more severe as heterogeneous modalities are added?

**Practical questions** (addressed by the prototype):

- **RQ4:** Within an audio-visual fusion system, does combining the joint cross-attention mechanism (which fuses a *combined* audio-visual representation back into the attention computation) with PDMP-style performance-dominant-modality gradient modulation outperform either technique applied alone, on a benchmark dataset such as CREMA-D or AffWild2?
- **RQ5:** How sensitive is fusion performance to the choice of which modality is treated as performance-dominant, given that the literature shows this can flip across datasets (audio dominant on AVE/VGGSound, visual dominant on CREMA-D) and is not always the optimization-dominant modality?

**The gap these survey papers leave unanswered:** none of the existing balanced-learning, dynamic-fusion, or low-quality-data surveys addresses modality fairness as a *design-time architectural question* (which signal classes are even eligible to enter the fusion pipeline) — they only address it as a *training-time optimization question* (how to weight modalities already present). This project treats both as necessary but separate problems.

---

## 5. Research Objectives

### Theoretical Objective
Formulate a unified six-modality framework (Language, Audio, Vision, Touch & Motion, Physiological Indicators, Environmental Sensors) that:
- Extends the data-centric taxonomy of multimodal challenges (noisy / incomplete / imbalanced / quality-varying, as defined in the comprehensive survey) to explicitly include physiological and environmental signal classes;
- Evaluates whether the performance-dominant-modality principle (PDMP) and inverse-variance weighting principle (ARL) are theoretically extensible beyond the two- and three-modality cases validated in the literature;
- Produces a conceptual modality-fairness checklist that the Nabra project's system design can be evaluated against.

### Practical Objective
Design, implement, and evaluate a functional audio-visual fusion prototype that:
- Implements a joint/bi-directional cross-attention fusion architecture informed by the two emotion-recognition papers reviewed;
- Incorporates a modality-imbalance mitigation strategy informed by PDMP and/or ARL;
- Is evaluated on a recognized benchmark (e.g., CREMA-D, AVE, or AffWild2) using Accuracy, F1-Score, and Macro-F1, with results reported against the unimodal and naive-concatenation baselines documented in the reviewed papers.

---

## 6. Methodology & Technical Approach

### 6.1 Theoretical framework construction
- Build the six-modality taxonomy by mapping each of the four low-quality-data challenge categories (noisy, incomplete, imbalanced, quality-varying) from the comprehensive survey onto Touch & Motion, Physiological Indicators, and Environmental Sensors, identifying where existing techniques (weighted fusion, uncertainty-aware dynamic fusion, gradient modulation) plausibly transfer and where they do not.
- Formally restate the PDMP optimization-dependency criterion and the ARL bias-variance decomposition in modality-agnostic notation, and analyze (without claiming to have run the experiment) where the two- and three-modality proofs would need new assumptions to extend to six heterogeneous modalities.

### 6.2 Prototype technical environment
- **Framework:** PyTorch
- **Pretrained backbones:** Hugging Face Transformers (for any language-adjacent or CLIP-based components, following the text-guided contrastive approach in the bi-directional cross-attention paper)
- **Visual preprocessing:** OpenCV
- **Audio preprocessing:** Librosa (spectrogram extraction, following the 257×299 / 257×1,004 spectrogram conventions used in PDMP and ARL for CREMA-D/AVE)

### 6.3 Fusion architecture
- **Primary fusion mechanism:** Cross-attention, drawing on (a) the joint cross-attention design, where a concatenated audio-visual representation is fed back into the attention computation rather than relying on raw unimodal features alone, and (b) the bi-directional cross-attention design, where audio and visual streams symmetrically query each other.
- **Imbalance mitigation:** Apply a PDMP-style modality-analysis stage (train unimodal models, rank performance, identify the performance-dominant modality) followed by asymmetric gradient modulation, rather than the conventional balanced-gradient approach. As a comparison condition, evaluate the ARL-style variance-based modulation coefficient as an alternative weighting scheme.
- **Temporal modeling:** A lightweight temporal module (Temporal Convolutional Network, as used for visual sequences in the bi-directional cross-attention paper) if the chosen benchmark and timeline permit sequence-level modeling rather than clip-level classification.

### 6.4 Evaluation metrics
- **Accuracy** and **Macro-F1**, matching the metrics reported across PDMP, ARL, and the comprehensive survey, to allow direct, apples-to-apples comparison against published baselines (e.g., PDMP's 80.21% accuracy / 80.34% macro-F1 on CREMA-D; ARL's 76.61% / 77.14% on the same dataset).
- Where feasible, **Concordance Correlation Coefficient (CCC)**, matching the dimensional emotion-recognition protocol used by both cross-attention papers, if the team adopts AffWild2 or a similar continuous valence/arousal benchmark instead of a discrete-emotion dataset.

---

## 7. Expected Results & Impact

**On the practical/empirical side:** Based on the magnitude of gains reported in the reviewed literature — PDMP improving over the strongest prior baseline by 6.25% (CREMA-D), 2.56% (Kinetics-Sounds), and 2.18% (AVE) accuracy, and ARL improving over its strongest baseline by roughly 3% accuracy on CREMA-D — we expect a prototype combining joint/bi-directional cross-attention with performance-dominant-modality gradient modulation to outperform a naive concatenation baseline by a comparable or larger margin, and to be competitive with (though not necessarily exceed) the specific numbers reported in PDMP and ARL, since this project does not have the same compute budget or hyperparameter search depth as those publications.

**On the theoretical side:** We expect the six-modality framework to surface concrete, citable gaps rather than to produce a fully validated new algorithm within the project timeline — specifically, a reasoned argument for why physiological and environmental signals cannot simply be slotted into the existing PDMP/ARL machinery without modification, grounded in the specific mathematical assumptions (e.g., the logit-based optimization-dependency coefficient in PDMP, Equation 5; the bias-variance decomposition in ARL, Equations 7–16) that those frameworks rely on.

**Broader impact:** The project's significance is twofold. Academically, it contributes a structured argument for modality fairness — naming and motivating a problem (the architectural exclusion of physiological/environmental signals) that the current balanced-multimodal-learning literature does not address, even as it solves an adjacent and more narrow problem (gradient-level imbalance between already-included modalities) with increasing sophistication. Practically, it directly informs the Nabra project's system design by giving the team a documented, citation-backed rationale for which sensing modalities deserve architectural consideration, and a working audio-visual fusion component that the team can extend.

---

## 8. Strict 20-Day Execution Timeline

| Phase | Days | Deliverables |
|---|---|---|
| **Data Synthesis & Theoretical Framework** | 1–5 | Consolidate the team's existing paper summaries; finalize the six-modality taxonomy and literature matrix; draft Sections 2–4 (Problem Statement, Literature Review, Research Questions) of the final paper. |
| **Prototype Development Phase** | 6–12 | Implement data preprocessing (Librosa/OpenCV pipelines); implement the joint/bi-directional cross-attention fusion architecture; implement PDMP-style or ARL-style imbalance mitigation; run initial training and baseline comparisons. |
| **Paper Drafting & Visualization Integration** | 13–17 | Write Methodology and Results sections; embed architecture diagrams and result figures from the team's "Figures & Architectures" directory; finalize quantitative comparison tables against PDMP/ARL/survey baselines. |
| **Final Review, Refinement, and Submission** | 18–20 | Internal review pass; refine framing of theoretical contributions and limitations; polish manuscript for presentation to the course professor. |

---

## 9. Research Significance

This project sits at the intersection of a genuine theoretical gap and a live industry need. Academically, it pushes back on an unexamined assumption running through even the most rigorous recent work on multimodal learning — that the relevant modality space is closed at audio, vision, and text — by naming physiological and environmental signals as modalities that deserve the same fairness scrutiny that PDMP and ARL have begun applying within the Traditional Triad, rather than treating them as noise to be filtered before fusion even begins. In doing so, it extends rather than contradicts the literature it builds on: PDMP and ARL's core insight, that imbalance should be measured and corrected rather than assumed away, is the same insight this project asks the field to apply to a wider set of signals. Directly, this research is built to feed the Nabra collaboration: a Saudi Arabian industry partner working in a sensor-rich domain stands to benefit concretely from a documented framework for evaluating which signal types belong in a fusion pipeline and how to weight them once included, backed by a working audio-visual prototype that demonstrates the state-of-the-art techniques in practice. Finally, the project has real-world utility beyond this course: as embedded and wearable sensing becomes cheaper and more pervasive, the systems that will benefit most from multimodal AI are precisely the ones the current literature has not yet been tested on — and a framework that anticipates this, even partially, has value beyond a single semester's deliverable.
