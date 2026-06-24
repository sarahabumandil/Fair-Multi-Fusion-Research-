# Multimodal Machine Learning — Exhaustive Technical Reference Manual

> **Tutorial Authors:** Louis-Philippe (LP) Morency (`morency@cs.cmu.edu`) & Tadas Baltrusaitis (`tbaltrus@cs.cmu.edu`)
> **Institution:** CMU Multimodal Communication and Machine Learning Laboratory — MultiComp Lab
> **Primary Reference:** Baltrusaitis, Ahuja, and Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, https://arxiv.org/abs/1705.09406
> **Scope:** This document is a zero-loss, exhaustive transcription and deep technical exposition of every slide and concept presented in the CMU/CVPR Tutorial on Multimodal Machine Learning (221 slides). No information has been omitted, summarized away, or compressed.

---

## Table of Contents

1. [Project Foundations: Architectural Overview & Theoretical Scope](#1-project-foundations-architectural-overview--theoretical-scope)
2. [Theoretical Framework: Modality vs. Medium](#2-theoretical-framework-modality-vs-medium)
3. [Examples of Modalities & Communicative Behaviors](#3-examples-of-modalities--multimodal-communicative-behaviors)
4. [Historical View: Four Eras of Multimodal Research](#4-historical-view-four-eras-of-multimodal-research)
5. [Real-World Multimodal Applications](#5-real-world-multimodal-applications)
6. [Core Technical Challenges in Multimodal Machine Learning](#6-core-technical-challenges-in-multimodal-machine-learning)
   - [Challenge 1: Representation Learning](#challenge-1-representation-learning)
   - [Challenge 2: Alignment](#challenge-2-alignment)
   - [Challenge 3: Fusion](#challenge-3-fusion)
   - [Challenge 4: Translation](#challenge-4-translation)
   - [Challenge 5: Co-Learning](#challenge-5-co-learning)
7. [Full Taxonomy of Multimodal Research](#7-full-taxonomy-of-multimodal-research)
8. [Basic Concepts — Part 1: Score & Loss Functions](#8-basic-concepts--part-1-score--loss-functions)
9. [Basic Concepts — Neural Networks](#9-basic-concepts--neural-networks)
10. [Basic Concepts — Optimization](#10-basic-concepts--optimization)
11. [Unimodal Representations: Language Modality](#11-unimodal-representations-language-modality)
12. [Unimodal Representations: Visual Modality & CNNs](#12-unimodal-representations-visual-modality--cnns)
13. [Unimodal Representations: Acoustic Modality & Autoencoders](#13-unimodal-representations-acoustic-modality--autoencoders)
14. [Multimodal Representations — Joint & Coordinated](#14-multimodal-representations--joint--coordinated)
15. [Basic Concepts — Part 2: Recurrent Neural Networks & LSTMs](#15-basic-concepts--part-2-recurrent-neural-networks--lstms)
16. [Translation and Alignment (Deep Dive)](#16-translation-and-alignment-deep-dive)
17. [Multimodal Fusion (Deep Dive)](#17-multimodal-fusion-deep-dive)
18. [System Architecture & Visual Pipeline](#18-system-architecture--visual-pipeline)

---

# 1. Project Foundations: Architectural Overview & Theoretical Scope

## Tutorial Schedule — Complete Syllabus

The tutorial is organized as a sequential body of knowledge spanning the following blocks:

**Block 1 — Introduction**
- What is Multimodal?
  - Historical view, multimodal vs. multimedia
- Why multimodal?
  - Multimodal applications: image captioning, video description, Audio-Visual Speech Recognition (AVSR), and others
- Core technical challenges
  - Representation learning, translation, alignment, fusion, and co-learning

**Block 2 — Basic Concepts, Part 1**
- Linear models
  - Score and loss functions, regularization
- Neural networks
  - Activation functions, multi-layer perceptron
- Optimization
  - Stochastic gradient descent, backpropagation

**Block 3 — Unimodal Representations**
- Visual representations
  - Convolutional neural networks
- Acoustic representations
  - Spectrograms, autoencoders

**Block 4 — Multimodal Representations**
- Joint representations
  - Visual semantic spaces, multimodal autoencoder
  - Tensor fusion representation
- Coordinated representations
  - Similarity metrics, canonical correlation analysis
- Coffee break [20 minutes]

**Block 5 — Basic Concepts, Part 2**
- Recurrent neural networks
  - Long Short-Term Memory models
- Optimization
  - Backpropagation through time

**Block 6 — Translation and Alignment**
- Translation applications
  - Machine translation, image captioning
- Explicit alignment
  - Dynamic time warping, deep canonical time warping
- Implicit alignment
  - Attention models, multi-instance learning
  - Temporal attention-gated model

**Block 7 — Multimodal Fusion**
- Model-free approaches
  - Early and late fusion, hybrid models
- Kernel-based fusion
  - Multiple kernel learning
- Multimodal graphical models
  - Factorial HMM, Multi-view Hidden CRF
  - Multi-view LSTM model

---

## 1.1 What is "Multimodal"? — Root Mathematical/Statistical Definition

The term **multimodal** originates from the statistical concept of a **multimodal distribution**.

![Multimodal Distribution Diagram](images/multimodal_distribution.jpg)

> A multimodal distribution is a probability distribution with **multiple modes** — that is, **distinct "peaks"** that appear as **local maxima** in the **probability density function (PDF)**. Each mode represents a region of high probability mass. In a unimodal distribution, there is exactly one such peak; in a bimodal distribution, two; in a multimodal distribution, two or more. The presence of multiple modes implies that the underlying data-generating process involves multiple distinct sub-populations or clusters.

In the context of machine learning and perception, the concept of "multimodal" is extended metaphorically: just as a multimodal PDF has multiple peaks representing distinct modes of variation, **multimodal machine learning** involves multiple distinct *types* (modes) of sensory information, each of which constitutes an independent data channel.

This foundational definition is critical: multimodal ML is not simply "using more than one feature type" — it is the study of architectures that meaningfully process and fuse information whose statistical structure, representation format, and measurement units are fundamentally heterogeneous.

---

# 2. Theoretical Framework: Modality vs. Medium

![Modality vs Medium Framework](images/modality_vs_medium.jpg)

> The diagram above illustrates the distinction between the *modality* (the information itself and how it is represented) and the *medium* (the physical or digital channel through which it is delivered). A failure to separate these two concepts historically caused confusion in the multimedia and multimodal research communities.

## 2.1 Formal Definition: Modality

**Modality** (as presented verbatim in the slides):

> *The way in which something happens or is experienced.*
>
> - **Modality** refers to a certain **type of information** and/or the **representation format** in which information is stored.
> - **Sensory modality**: one of the primary forms of sensation, as vision or touch; **channel of communication**.

A modality therefore has two defining attributes:
1. **Content** — what type of information it carries (visual, auditory, linguistic, haptic, etc.)
2. **Format** — how that information is encoded and stored (pixel arrays, waveforms, token sequences, point clouds, etc.)

## 2.2 Formal Definition: Medium

**Medium** (as presented verbatim in the slides):

> *A means or instrumentality for storing or communicating information; system of communication/transmission.*
>
> - **Medium** is the means whereby this information is **delivered to the senses of the interpreter**.

The medium answers "through what channel is the information transmitted?" — e.g., television, the internet, physical air (sound waves), paper, or a display screen. The medium is infrastructure; the modality is content.

## 2.3 Paradigm Shift: Multimodal vs. Multimedia

The tutorial explicitly draws a historical distinction between **multimodal** and **multimedia** research communities:

- **Multimedia** (dominant 1990s–2000s framing): Focused on the storage, retrieval, and delivery of heterogeneous content types (video, audio, text) as *media artifacts*. The emphasis was on engineering systems that could handle different *media formats*.

- **Multimodal** (dominant framing since 2010s): Focuses on the *sensory and cognitive integration* of information from fundamentally different modalities. The emphasis is on learning *joint representations*, performing *cross-modal alignment*, and enabling *fusion* for downstream prediction tasks.

The key conceptual shift is from **format-level heterogeneity** (multimedia) to **semantic-level heterogeneity** (multimodal). Multimodal ML asks: how can a learning system jointly reason over signals that have completely different statistical structures, dimensionalities, and semantic grounding?

---

# 3. Examples of Modalities & Multimodal Communicative Behaviors

## 3.1 Complete Enumeration of Modalities (from slides)

The tutorial enumerates the following complete list of modalities considered in multimodal ML research:

1. **Natural language** — both spoken and written
2. **Visual** — from images or videos
3. **Auditory** — including voice, sounds, and music
4. **Haptics / touch**
5. **Smell, taste, and self-motion**
6. **Physiological signals**
   - Electrocardiogram (ECG)
   - Skin conductance
7. **Other modalities**
   - Infrared images
   - Depth images
   - fMRI (functional magnetic resonance imaging)

## 3.2 Multimodal Communicative Behaviors (Complete Taxonomy from Slides)

Human communication is inherently multimodal. The tutorial provides the following exhaustive taxonomy:

**Verbal Behaviors:**
- Lexicon
  - Words
- Syntax
  - Part-of-speech
  - Dependencies
- Pragmatics
  - Discourse acts

**Vocal Behaviors:**
- Prosody
  - Intonation
  - Voice quality
- Vocal expressions
  - Laughter, moans

**Visual Behaviors:**
- Gestures
  - Head gestures
  - Eye gestures
  - Arm gestures
- Body language
  - Body posture
  - Proxemics
- Eye contact
  - Head gaze
  - Eye gaze
- Facial expressions
  - FACS (Facial Action Coding System) action units
  - Smile, frowning

## 3.3 Multiple Communities and Modalities

The tutorial acknowledges that multimodal ML draws from and serves multiple research communities simultaneously:

- Psychology
- Medical
- Speech
- Vision
- Language
- Multimedia
- Robotics
- Learning (Machine Learning)

This cross-community nature explains why multimodal ML has historically lacked unified terminology and why bridging these communities is a key contribution of the tutorial and the associated survey paper.

---

# 4. Historical View: Four Eras of Multimodal Research

![Historical Timeline of Multimodal Research](images/historical_timeline.jpg)

> The timeline spans from 1970 to the present day, with four distinct eras demarcated by their dominant paradigm, methodology, and application focus. Each era built on the foundations of the previous while introducing fundamentally new capabilities.

## 4.1 Era 1: The "Behavioral" Era (1970s until late 1980s)

**Dominant paradigm:** Psychological and behavioral theory; multi-sensory integration studied experimentally.

**Key works and contributions:**
- **Multimodal Behavior Therapy** by Arnold Lazarus [1973] — proposed 7 dimensions of personality (or modalities), establishing an early vocabulary for multimodality in human behavioral science.

**Multi-sensory integration (in psychology):**
- *Multimodal signal detection: Independent decisions vs. integration* [1980] — studied whether humans combine sensory signals independently or integrate them.
- *Infants' perception of substance and temporal synchrony in multimodal events* [1983] — investigated cross-modal perception from birth.
- *A multimodal assessment of behavioral and cognitive deficits in abused and neglected preschoolers* [1984] — applied multimodal assessment methodology to clinical psychology.

**Landmark figure — David McNeill (University of Chicago, Center for Gesture and Speech Research):**
> *"For McNeill, gestures are in effect the speaker's thought in action, and integral components of speech, not merely accompaniments or additions."*

This was a pivotal conceptual contribution: McNeill demonstrated that gesture and speech are not independent channels but deeply intertwined cognitive and communicative processes.

**The McGurk Effect [1976] — "Hearing lips and seeing voices" (Nature):**

![McGurk Effect Diagram](images/mcgurk_effect.jpg)

> The McGurk Effect, published in *Nature* in 1976, is one of the most influential empirical demonstrations of multimodal integration in human perception. When a subject watches a video of lips forming the syllable "ga" while hearing the sound "ba," they perceive neither "ba" nor "ga" but an entirely new percept "da." This demonstrates that the auditory and visual speech streams are not processed independently in the brain but are fused at a preconscious level into a unified perceptual experience. The McGurk Effect became the motivating biological phenomenon for Audio-Visual Speech Recognition (AVSR) research.

**Trivia:** Geoffrey Hinton received his B.A. in Psychology — illustrating the deep roots of neural computation in cognitive and behavioral science.

## 4.2 Era 2: The "Computational" Era (Late 1980s until 2000)

**Dominant paradigm:** Engineering computational systems that process multiple input channels. Three main thrusts:

### Thrust 1: Audio-Visual Speech Recognition (AVSR)
- Directly motivated by the McGurk effect.
- **First AVSR System in 1986:** *"Automatic lipreading to enhance speech recognition"* — pioneered the use of lip reading as a complementary signal to acoustic speech recognition.
- **Good survey paper [2002]:** *"Recent Advances in the Automatic Recognition of Audio-Visual Speech"* — consolidated a decade of AVSR progress.
- **Trivia:** The first multimodal deep learning paper was about audio-visual speech recognition [ICML 2011].

### Thrust 2: Multimodal/Multisensory Interfaces (Human-Computer Interaction)
> *"Study of how to design and evaluate new computer systems where humans interact through multiple modalities, including both input and output modalities."*

- **Glove-talk:** A neural network interface between a data-glove and a speech synthesizer, by Sidney Fels & Geoffrey Hinton [CHI'95] — demonstrated a learned mapping from gestural input (a data glove) to speech synthesis output.
- **Affective Computing — Rosalind Picard:**
  > *"Affective Computing is computing that relates to, arises from, or deliberately influences emotion or other affective phenomena."*
  This laid the groundwork for all subsequent work on multimodal emotion recognition.

### Thrust 3: Multimedia Computing
- **The Informedia Digital Video Library Project [1994–2010]:**
  > *"The Informedia Digital Video Library Project automatically combines speech, image and natural language understanding to create a full-content searchable digital video library."*

  **Multimedia content analysis tasks pioneered:**
  - **Shot-boundary detection (1991–):** Parsing a video into continuous camera shots.
  - **Still and dynamic video abstracts (1992–):**
    - Making video browsable via representative frames (keyframes).
    - Generating short clips carrying the essence of the video content.
  - **High-level parsing (1997–):** Parsing a video into semantically meaningful segments.
  - **Automatic annotation/indexing (1999–):** Detecting prespecified events, scenes, and objects in video.

### Computational Models of the Computational Era

**Hidden Markov Models [1960s]:**

![Hidden Markov Model Diagram](images/hmm_diagram.jpg)

> An HMM is a generative probabilistic model where an observed sequence x₁, x₂, ..., xₙ is assumed to be generated by an underlying hidden state sequence h₀, h₁, h₂, h₃, h₄. The hidden states form a Markov chain (each state depends only on the previous), while observations are conditionally independent given the hidden state. HMMs were foundational to both speech recognition and multimodal temporal modeling.

- **Factorial Hidden Markov Models [1996]:** Extended HMMs by assuming multiple independent hidden chains generating the observed data simultaneously — a natural fit for multimodal signals.
- **Coupled Hidden Markov Models [1997]:** Further extended HMMs to model cross-chain dependencies, allowing one modality's hidden state sequence to influence another's.

**Artificial Neural Networks [1940s]:**
- **Backpropagation [1975]:** The training algorithm enabling gradient-based learning in multi-layer networks.
- **Convolutional Neural Networks [1980s]:** Introduced weight sharing and spatial locality into neural architectures, revolutionizing visual processing.

## 4.3 Era 3: The "Interaction" Era (2000s)

**Dominant paradigm:** Modeling naturalistic human-human and human-computer interaction in real-world, multi-sensor settings.

### Major Projects:

**Modeling Human Multimodal Interaction:**
- **AMI Project [2001–2006, IDIAP]:**
  - 100+ hours of meeting recordings
  - Fully synchronized audio-video
  - Transcribed and annotated
  - *Trivia: Samy Bengio started at IDIAP working on the AMI project.*

- **CHIL Project [Alex Waibel]:**
  - Computers in the Human Interaction Loop
  - Multi-sensor multimodal processing
  - Face-to-face interactions

- **CALO Project [2003–2008, SRI]:**
  - Cognitive Assistant that Learns and Organizes
  - Personalized Assistant that Learns (PAL)
  - *Siri was a spinoff from this project.*
  - *Trivia: LP's PhD research was partially funded by CALO.*

- **SSP Project [2008–2011, IDIAP]:**
  - Social Signal Processing
  - First coined by Sandy Pentland in 2007
  - Dataset repository: http://sspnet.eu/

**Multimedia Information Retrieval:**
- **TRECVid [Hosted by NIST, 2001–2016]:**
  > *"Yearly competition to promote progress in content-based retrieval from digital video via open, metrics-based evaluation."*

  Research tasks:
  - Shot boundary, story segmentation, search
  - "High-level feature extraction": semantic event detection
  - Introduced in 2008: copy detection and surveillance events
  - Introduced in 2010: Multimedia event detection (MED)

### Computational Models of the Interaction Era:

**Dynamic Bayesian Networks:**
- Kevin Murphy's PhD thesis and Matlab toolbox
- Asynchronous HMM for multimodal [Samy Bengio, 2007]
- Applied to audio-visual speech segmentation

**Discriminative Sequential Models:**
- **Conditional Random Fields [Lafferty et al., 2001]:** Discriminative sequence models that directly model the conditional distribution P(y|x), avoiding the generative assumptions of HMMs.
- **Latent-dynamic CRF [Morency et al., 2007]:** Extended CRFs with latent variables to capture unobserved, internal dynamic structure.

## 4.4 Era 4: The "Deep Learning" Era (2010s until present)

> **This era is the primary focus of the tutorial.**

**Representation learning (deep learning) milestones:**
- **Multimodal deep learning [ICML 2011]:** First paper to apply deep representation learning to multimodal data.
- **Multimodal Learning with Deep Boltzmann Machines [NIPS 2012]:** Generative deep model for joint multimodal representation.
- **Visual attention — Show, Attend and Tell: Neural Image Caption Generation with Visual Attention [ICML 2015]:** Introduced soft and hard attention mechanisms for grounding linguistic outputs in spatial visual features.

**Key enablers for multimodal research in the deep learning era:**
1. New large-scale multimodal datasets
2. Faster computers and GPUs
3. High-level visual features (from pre-trained CNNs)
4. "Dimensional" linguistic features (word embeddings, word2vec)

**New challenges and multimodal corpora:**
- **Audio-Visual Emotion Challenge (AVEC, 2011–):**
  - Emotional dimension estimation
  - Standardized training and test sets
  - Based on the SEMAINE dataset

- **Emotion Recognition in the Wild Challenge (EmotiW 2013–):**
  - Emotional dimension estimation
  - Standardized training and test sets
  - Based on the SEMAINE dataset

**Renewal of multimedia content analysis:**
- Image captioning
- Video description
- Visual Question-Answer (VQA)

---

# 5. Real-World Multimodal Applications

The tutorial enumerates the following complete taxonomy of real-world tasks tackled by multimodal ML research:

**Affect Recognition:**
- Emotion recognition
- Persuasion detection
- Personality trait inference

**Media Description:**
- Image captioning
- Video captioning
- Visual Question Answering (VQA)

**Event Recognition:**
- Action recognition
- Action segmentation

**Multimedia Information Retrieval:**
- Content-based retrieval
- Cross-media retrieval

---

# 6. Core Technical Challenges in Multimodal Machine Learning

![Core Challenges Overview](images/core_challenges_overview.jpg)

> The five core challenges — Representation, Alignment, Fusion, Translation, and Co-Learning — are **non-exclusive**. A single multimodal system may simultaneously require solutions to all five. Their interdependencies mean that architectural decisions made for fusion will impact alignment, and representation choices constrain what translation approaches are feasible.

**Source paper:** Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippe Morency. *Multimodal Machine Learning: A Survey and Taxonomy.* https://arxiv.org/abs/1705.09406

---

## Challenge 1: Representation Learning

> **Definition:** Learning how to represent and summarize multimodal data in a way that **exploits the complementarity and redundancy** of multiple modalities.

![Representation Learning Architecture](images/representation_challenge.jpg)

> The diagram shows the two fundamental paradigms: (A) Joint Representations, where all modalities are mapped into a single shared space; and (B) Coordinated Representations, where each modality maintains its own representation space but these spaces are brought into structured correspondence via a similarity or structural constraint.

### Sub-Type A: Joint Multimodal Representations

In joint representation learning, a single unified representation vector is computed from all input modalities. The architecture typically consists of unimodal encoders feeding into a shared latent space.

**Motivating example — Joint Space Semantics:**
- Input: Text *"I like it!"* + Joyful tone → mapped to a joint multimodal space
- Input: Text *"Wow!"* + Tensed voice → mapped to a nearby or distinct region of the same joint space

The joint space allows semantic similarity to be measured via distance metrics (cosine similarity, Euclidean distance) regardless of the original modality.

**Key architectures for Joint Representations:**

**1. Bimodal Deep Belief Network [Ngiam et al., ICML 2011]:**
- Applied to audio-visual speech recognition
- Trained a generative model with both audio and visual inputs
- Showed that the joint representation outperformed unimodal representations, even when only one modality was available at test time

**2. Multimodal Deep Boltzmann Machine [Srivastava and Salakhutdinov, NIPS 2012]:**
- A generative model with individual modality towers (for depth, video, verbal/text) feeding into a shared top-level representation
- Used for image tagging and cross-media retrieval
- Allowed reconstruction of one modality from another

**3. Deep Boltzmann Machine for Audio-Visual Emotion Recognition [Kim et al., ICASSP 2013]:**
- Separate DBM towers for visual and verbal/text features
- Merged at the joint representation layer

**Multimodal Vector Space Arithmetic:**

![Multimodal Vector Space Arithmetic](images/joint_representation.jpg)

> [Kiros et al., *Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models*, 2014] demonstrated that once visual and linguistic features are embedded in a joint space, algebraic operations become semantically meaningful. Image vectors can be added to or subtracted from word vectors to navigate the joint semantic space in interpretable ways.

**Multimodal Sentiment Analysis — Joint Representations:**

![Multimodal Sentiment Architecture](images/joint_vs_coordinated.jpg)

> For sentiment intensity prediction from text (X), image/video (Y), and audio (Z), a joint representation is formed as:
> **h_m = f(W · [h_x, h_y, h_z])**
> where each unimodal representation h_x, h_y, h_z is computed by independent unimodal encoder stacks, and their concatenation is passed through a fully connected fusion layer to a softmax output over sentiment intensity [-3, +3].

**Unimodal, Bimodal, and Trimodal Interactions — Motivating Example:**

The following examples from the slides illustrate *why* modeling higher-order multimodal interactions matters:

| Speaker behavior | Interpretation |
|---|---|
| "This movie is sick" (text only) | Ambiguous — sick = great OR awful |
| "This movie is fair" (text only) | Mildly negative — unimodal cue |
| Smile (visual only) | Positive |
| Loud voice (acoustic only) | Ambiguous |
| "This movie is sick" + Smile | Bimodal: confirms positive |
| "This movie is sick" + Frown | Bimodal: confirms negative |
| "This movie is sick" + Loud voice | Still ambiguous — trimodal needed |
| "This movie is sick" + Smile + Loud voice | Different trimodal interaction |
| "This movie is fair" + Smile + Loud voice | Different trimodal interaction |

This motivates explicit modeling of unimodal, bimodal, and trimodal interaction terms.

**Multimodal Tensor Fusion Network (TFN) [Zadeh, Jones, and Morency, EMNLP 2017]:**

![Tensor Fusion Network Diagram](images/tensor_fusion_network.jpg)

> The Tensor Fusion Network explicitly models both unimodal and bimodal interactions via an outer product (tensor product) operation. For two modalities:
>
> **h_m = [h_x; 1] ⊗ [h_y; 1] = [h_x⊗h_y; h_x; h_y; 1]**
>
> The appended `1` scalar ensures that the outer product captures unimodal terms (h_x, h_y), the bimodal interaction term (h_x⊗h_y), and a constant bias term. For three modalities (text X, image Y, audio Z):
>
> **h_m = [h_x; 1] ⊗ [h_y; 1] ⊗ [h_z; 1]**
>
> This produces all combinations: unimodal (h_x, h_y, h_z), bimodal (h_x⊗h_y, h_x⊗h_z, h_z⊗h_y), and trimodal (h_x⊗h_y⊗h_z) interaction terms simultaneously. The full tensor is then passed to a fusion layer and softmax for prediction. Results on the MOSI dataset showed improvement over state-of-the-art in multimodal sentiment analysis.

**Multimodal Encoder-Decoder for Image Captioning:**

![Encoder-Decoder Architecture](images/attention_image_captioning.jpg)

> The encoder-decoder architecture for image captioning couples a CNN visual encoder with an LSTM language decoder. The CNN computes a fixed-dimensional visual representation from the image (Y). A simple multi-layer perceptron translates from CNN visual space to LSTM input space. The LSTM then decodes the visual representation into a natural language caption sequence (text X). This architecture generalizes to video description by replacing the image CNN with a video-level temporal feature aggregator.

### Sub-Type B: Coordinated Multimodal Representations

In coordinated representation learning, each modality maintains its own dedicated representation space, but these spaces are aligned or coordinated via a loss function that enforces structural constraints — typically similarity or correlation-based.

**Coordinated Multimodal Embeddings:**

![Coordinated Representation Framework](images/joint_vs_coordinated.jpg)

> [Huang et al., *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data*, 2013] showed that separate modality encoders can be trained such that matched (paired) instances from different modalities are mapped to nearby points, while unmatched instances are mapped far apart. A cosine similarity metric is used to measure correspondence.

**Canonical Correlation Analysis (CCA) — Deep Dive:**

CCA is the classical algorithm for learning coordinated representations. The slides present it in full mathematical detail:

**Step 1 — Learn two linear projections (one per view) that are maximally correlated:**

Maximize corr(u^T X, v^T Y) ≈ u^T Σ_XY v  →  optimal u*, v*

**Step 2 — Learn multiple projection pairs that are mutually orthogonal ("canonical"):**

Maximize corr(u^T_(i) X, v^T_(i) Y) for each pair i, subject to orthogonality:
u^T_(i) Σ_XY v^(j) = u^T_(j) Σ_XY v^(i) = 0 for i ≠ j

Combined objective: maximize tr(U Σ_XY V)
where U = [u^(1), ..., u^(k)] and V = [v^(1), ..., v^(k)]

**Step 3 — Constrain projections to unit variance:**
U^T Σ_XX U = I,  V^T Σ_YY V = I

**Full CCA optimization problem:**
- Maximize: tr(U^T Σ_XY V)
- Subject to: U^T Σ_XX U = V^T Σ_YY V = I

This is solved via eigen-decomposition of the combined covariance matrix Σ, producing projection pairs U, V.

**Important equivalence:** When data is normalized, CCA is equivalent to smallest RMSE reconstruction. The CCA loss can be rewritten as:

L(U, V) = ||U^T X - V^T Y||^2_F

subject to: U^T Σ_XX U = V^T Σ_YY V = I

**Deep Canonical Correlation Analysis (DCCA) [Andrew et al., ICML 2013]:**

![Deep CCA Architecture](images/deep_cca_arch.jpg)

> DCCA replaces the linear projections of CCA with deep non-linear networks. The objective remains identical to CCA:
>
> argmax_{V,U,W_x,W_y} corr(H_x, H_y)
>
> where H_x = W_x^(deep)(X) and H_y = W_y^(deep)(Y) are the outputs of deep non-linear encoders for text and image respectively, followed by linear projection layers U and V. The deep architecture allows capturing non-linear cross-modal correlations that linear CCA cannot discover.
>
> **Three constraints are preserved from classical CCA:**
> 1. Linear projections maximizing correlation
> 2. Orthogonal projections
> 3. Unit variance of the projection vectors

**Deep Canonically Correlated Autoencoders (DCCAE) [Wang et al., ICML 2015]:**

DCCAE extends DCCA by jointly optimizing for both DCCA correlation and autoencoder reconstruction:

![DCCAE Architecture](images/dccae.jpg)

> DCCAE jointly optimizes two objectives simultaneously: (1) maximize cross-view correlation (DCCA objective) between the shared representation spaces H_x and H_y; and (2) minimize reconstruction error for each individual view (autoencoder objective), where decoders reconstruct X' from H_x and Y' from H_y. There is an explicit trade-off between multi-view correlation and reconstruction error from individual views, controlled by a hyperparameter λ. This trade-off allows DCCAE to learn representations that are both cross-modally coherent and individually informative.

---

## Challenge 2: Alignment

> **Definition:** Identify the **direct relations** between (sub)elements from two or more **different modalities**.

![Alignment Problem Diagram](images/alignment_challenge.jpg)

> The alignment problem involves finding correspondences between temporally or structurally heterogeneous multimodal data streams. For example, given a spoken sentence and a sequence of gestures, which gesture segment corresponds to which word or phrase? Elements of Modality 1 (t₁, t₂, t₃, ..., tₙ) and Modality 2 (t₄, t₅, ..., tₙ) are connected by a correspondence-finding algorithm.

Two major sub-types are defined:

### Sub-Type A: Explicit Alignment
> The goal is to **directly find correspondences** between elements of different modalities.

### Sub-Type B: Implicit Alignment
> Uses **internally latent alignment** of modalities in order to **better solve a different problem** (e.g., image captioning).

A full deep-dive into alignment methods is presented in Section 16.

---

## Challenge 3: Fusion

> **Definition:** To join information from two or more modalities to **perform a prediction task**.

![Fusion Taxonomy Overview](images/fusion_early_late.jpg)

> Fusion is the central operation in multimodal systems. The diagram shows the high-level taxonomy: model-agnostic approaches (early and late fusion) vs. model-based approaches (deep neural networks, kernel methods, graphical models). The choice of fusion strategy has profound consequences for what cross-modal interactions can be captured, how the system handles missing modalities, and its computational cost.

This challenge is explored in exhaustive depth in Section 17.

---

## Challenge 4: Translation

> **Definition:** Process of **changing data from one modality to another**, where the translation relationship can often be **open-ended or subjective**.

![Translation Challenge Diagram](images/translation_challenge.jpg)

> Translation is fundamentally different from alignment: alignment finds correspondences between existing multimodal streams, while translation generates new content in one modality given input from another. The open-ended nature of translation makes it particularly challenging to evaluate — unlike alignment (where ground truth correspondences exist), translation outputs may have many equally valid correct answers.

**Two major categories:**
- **A: Example-based** — retrieving or combining existing examples from a database
- **B: Model-driven** — learning a parametric generative model for translation

**Translation challenges (complete list from slides):**
1. Different representations across modalities
2. Multiple source modalities
3. Open-ended translations (many valid outputs)
4. Subjective evaluation
5. Repetitive processes (e.g., repeated gestures in animation synthesis)

**Translation application domains:**
- Visual animations
- Image captioning
- Speech synthesis

A full deep-dive into translation methods is presented in Section 16.

---

## Challenge 5: Co-Learning

> **Definition:** Transfer knowledge between modalities, including their **representations and predictive models**.

![Co-Learning Framework](images/co_learning.jpg)

> Co-learning addresses the scenario where one modality is used to help train a model for another modality during training. The "help during training" arrow indicates that the secondary modality acts as a supervision or regularization signal during the learning of the primary modality's model. At inference time, only one modality may be needed.

**Three sub-types of Co-Learning:**

**A: Parallel Data**
- Both modalities are available for the same instances
- Methods: co-training, transfer learning

**B: Non-Parallel Data**
- Instances may not be paired across modalities
- Methods:
  - Zero-shot learning
  - Concept grounding
  - Transfer learning

**C: Hybrid Data**
- A combination of parallel and non-parallel data
- Methods: Bridging (using a third modality or dataset as a bridge between two modalities that cannot be directly paired)

---

# 7. Full Taxonomy of Multimodal Research

The complete taxonomy from [Baltrusaitis, Ahuja, and Morency, https://arxiv.org/abs/1705.09406]:

```
Representation
  ├─ Joint
  │    ├─ Neural networks
  │    ├─ Graphical models
  │    └─ Sequential
  └─ Coordinated
       ├─ Similarity
       └─ Structured

Translation
  ├─ Example-based
  │    ├─ Retrieval
  │    └─ Combination
  └─ Model-based
       ├─ Grammar-based
       ├─ Encoder-decoder
       └─ Online prediction

Alignment
  ├─ Explicit
  │    ├─ Unsupervised
  │    └─ Supervised
  └─ Implicit
       ├─ Graphical models
       └─ Neural networks

Fusion
  ├─ Model agnostic
  │    ├─ Early fusion
  │    ├─ Late fusion
  │    └─ Hybrid fusion
  └─ Model-based
       ├─ Kernel-based
       ├─ Graphical models
       └─ Neural networks

Co-learning
  ├─ Parallel data
  │    ├─ Co-training
  │    └─ Transfer learning
  ├─ Non-parallel data
  │    ├─ Zero-shot learning
  │    ├─ Concept grounding
  │    └─ Transfer learning
  └─ Hybrid data
       └─ Bridging
```

---

# 8. Basic Concepts — Part 1: Score & Loss Functions

## 8.1 Linear Classification

A linear classifier operates in three steps:
1. Define a (linear) score function
2. Define the loss function (possibly non-linear)
3. Optimization

### 8.1.1 Score Function

For a multi-class linear classifier with input observation x_i (e.g., a 32×32×3 image flattened to a [3072×1] vector):

**f(x_i; W, b) = Wx_i + b**

Where:
- x_i is the input observation [3072×1] (ith element of the dataset)
- W is the weight matrix [10×3072]
- b is the bias vector [10×1]
- f(x_i; W, b) is the class score vector [10×1]
- Combined parameters: [10×3073]

The linear classifier defines a planar decision surface in data-space: Wx_i + b > 0 (with w as the normal vector and -b/||w|| as the offset from the origin).

**Notation tricks — Bias absorption:**
By appending a "1" to the end of each input observation vector x_i (making it [3073×1]), the bias vector becomes the last column of the weight matrix W (now [10×3073]):

**f(x_i; W) = Wx_i**

**Label-specific notation:**
- "dog" linear classifier: f(x_i; W_dog, b_dog) or f(x_i; W, b)_dog or f_dog
- Linear classifier for label j: f(x_i; W_j, b_j) or f(x_i; W, b)_j or f_j

### 8.1.2 Loss Functions

**Loss Function (general):** The loss function quantifies the amount by which the prediction scores deviate from the actual values.

#### Loss Function 1: Cross-Entropy Loss (Logistic Loss)

**Logistic (sigmoid) function:**

σ(f) = 1 / (1 + e^{-f})

Maps any real score f to the range (0, 1). At f=0, σ(f) = 0.5.

**Logistic regression (two-class problem):**

p(y_i = "dog" | x_i; w) = σ(w^T x_i)

**Softmax function (multiple classes):**

p(y_i | x_i; W) = e^{f_{y_i}} / Σ_j e^{f_j}

**Cross-entropy loss:**

L_i = -log [ e^{f_{y_i}} / Σ_j e^{f_j} ]

This minimizes the **negative log likelihood** of the correct class label. Equivalently, it maximizes the log probability of the correct class.

#### Loss Function 2: Hinge Loss (Max-Margin Loss / Multi-class SVM Loss)

L_i = Σ_{j ≠ y_i} max(0, f_j - f_{y_i} + Δ)

Where Δ is the margin (e.g., Δ = 10).

- Loss due to example i: sum over all incorrect labels
- Difference between the correct class score and incorrect class score
- Loss is zero when f_{y_i} exceeds all f_j by at least margin Δ

**Full loss across all training samples:**

L = Σ_i L_i

---

# 9. Basic Concepts — Neural Networks

## 9.1 Neural Network Inspiration

Neural networks are made up of **artificial neurons**, inspired by biological neurons in the brain.

## 9.2 Basic Neural Network Building Block

Each artificial neuron computes a **weighted sum** followed by a **nonlinear activation function**:

**y = f(Wx + b)**

Where:
- Input: x
- Weighted sum: Wx + b
- Activation function: f(·)
- Output: y

## 9.3 Activation Functions

Four primary activation functions are detailed in the slides:

1. **Hyperbolic tangent:** f(x) = tanh(x) — outputs in range (-1, 1)
2. **Sigmoid:** f(x) = (1 + e^{-x})^{-1} — outputs in range (0, 1)
3. **Linear:** f(x) = ax + b — no nonlinearity (identity)
4. **ReLU (Rectified Linear Unit):** f(x) = max(0, x) ≈ log(1 + exp(x))
   - **Faster training** — no gradient vanishing in the positive region
   - **Induces sparsity** — negative inputs produce zero output, leading to sparse activations

## 9.4 Multi-Layer Feedforward Network

The score function for a 3-layer network:

y_i = f(x_i) = f_{3;W_3}(f_{2;W_2}(f_{1;W_1}(x_i)))

Activation functions for individual layers:
- f_{1;W_1}(x) = σ(W_1 x + b_1)
- f_{2;W_2}(x) = σ(W_2 x + b_2)
- f_{3;W_3}(x) = σ(W_3 x + b_3)

**Loss function (Euclidean loss for data-point i):**

L_i = (f(x_i) - y_i)^2

**Full loss across all training samples:**

L = Σ_i (f(x_i) - y_i)^2

---

# 10. Basic Concepts — Optimization

## 10.1 Gradient Concept

**Geometrically:** The gradient points in the direction of the greatest rate of increase of the function, with magnitude equal to the slope of the graph in that direction.

**In 1D:**

df(x)/dx = lim_{h→0} [f(x+h) - f(x)] / h

**In higher dimensions (partial derivative):**

∂f/∂x_i (a_1, ..., a_n) = lim_{h→0} [f(a_1, ..., a_i+h, ..., a_n) - f(a_1, ..., a_i, ..., a_n)] / h

In multiple dimensions, the gradient is the **vector of partial derivatives** (called the **Jacobian**).

## 10.2 Gradient Computation — Chain Rule

**Single path:**

∂y/∂x = (∂y/∂h) · (∂h/∂x)    where y = f(h) and h = g(x)

**Multiple path chain rule:**

∂y/∂x = Σ_j (∂y/∂h_j) · (∂h_j/∂x)    where y = f(h_1, h_2, h_3) and each h_j = g(x)

**Multiple inputs:**

∂y/∂x_1 = Σ_j (∂y/∂h_j) · (∂h_j/∂x_1)

**Vector representation (gradient):**

∇_x y = (∂y/∂x_1, ∂y/∂x_2, ∂y/∂x_3)

∇_x y = (∂h^T/∂x) ∇_h y

where (∂h^T/∂x) is the "local" Jacobian (matrix of size |h| × |x|, computed using partial derivatives) and ∇_h y is the "backprop" gradient.

## 10.3 Backpropagation Algorithm

**Forward pass:** Following the graph topology, compute the value of each unit.

**Backpropagation pass:**
1. Initialize output gradient = 1
2. Compute "local" Jacobian matrix using values from forward pass
3. Apply chain rule: Gradient = "local" Jacobian × "backprop" gradient

**Example network (3-layer):**
- L = -log P(Y = y | z)  (cross-entropy)
- z = matmult(h_2, W_3)
- h_2 = f(h_1; W_2)
- h_1 = f(x; W_1)

## 10.4 Optimization Methods

**Gradient Descent:**

θ^{(t+1)} = θ^{(t)} - ε_k ∇_θ L

Where:
- θ^{(t+1)} — new model parameters
- θ^{(t)} — previous model parameters
- ε_k — learning rate at iteration k

**Learning rate decay (linear schedule until iteration τ):**

ε_k = (1 - α)ε_0 + α ε_τ

**Extensions to basic gradient descent:**
- Stochastic ("batch") gradient descent
- With momentum
- AdaGrad
- RMSProp

**Other optimization methods:**
- **Newton methods** — use Hessian (second derivative)
- **Quasi-Newton methods** — use approximate Hessian:
  - BFGS
  - LBFGS
  - Do not require learning rates (fewer hyperparameters)
  - Do not work with stochastic and batch methods — rarely used to train modern Neural Networks

---

# 11. Unimodal Representations: Language Modality

## 11.1 Word-Level Representations

For **word-level** classification (part-of-speech, named entity recognition):

**Input representation:** one-hot vector x_i of dimension equal to the number of words in the dictionary.

x_i = [0; 0; ...; 0; 1; 0; ...; 0]

Only the entry corresponding to the current word is 1; all others are 0.

**Output classifications:**
- Part-of-speech (noun, verb, ...)
- Named entity (names of persons, ...)

## 11.2 Document-Level Representations

For **document-level** classification (sentiment analysis):

**Input representation:** bag-of-words vector x_i of dimension equal to the number of words in the dictionary. Each entry counts the number of times a word appears in the document.

## 11.3 Learning Better Language Representations: Word Embeddings

**Distribution hypothesis:** *Approximate word meaning by its surrounding words.*
*"Words used in a similar context will lie close together."*

Example: "He was walking away because..." and "He was running away because..." share nearly identical contexts, suggesting "walking" and "running" should be semantically close.

**Word2vec approach:** Instead of capturing co-occurrence counts directly, predict surrounding words of every word.

**Architecture:**
- Input: one-hot vector [100,000-dimensional] for the target word (e.g., "walking")
- No activation function → very fast
- Hidden layer: 300-dimensional dense embedding
- Output: one-hot vectors for surrounding words (He, was, away, because, ...)

[100,000-dim one-hot] × W_1 → [300-dim embedding] × W_2 → [100,000-dim output]

**Word2vec algorithm:** https://code.google.com/p/word2vec/

## 11.4 Benefits of Word Embeddings vs. One-Hot

If vocabulary = 100,000 words:

**Classic NLP (one-hot):**
- Walking: [0; 0; ...; 1; ...; 0] (100,000-dimensional)
- Running: [0; 0; ...; 0; ...; 1] (100,000-dimensional)
- Similarity = 0.0 (orthogonal by design)

**Word embeddings (after training):**
- Walking: [0.1; 0.0003; 0; ...; 0.02; 0.08; 0.05] (300-dimensional)
- Running: [0.1; 0.0004; 0; ...; 0.01; 0.09; 0.05] (300-dimensional)
- Similarity = 0.9 (similar context → similar embedding)

## 11.5 Vector Space Models of Words

A vector space is built in which all words reside with certain relationships between them. This vector space encodes both **syntactic** and **semantic** relationships, allowing algebraic operations:

vec(king) - vec(man) + vec(woman) ≈ vec(queen)

The vector space encodes semantic analogies (king:queen :: man:woman) and syntactic regularities (walk:walked :: run:ran).

**Training scale:** Trained on the Google News corpus with over 300 billion words. [Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality", NIPS 2013]

---

# 12. Unimodal Representations: Visual Modality & CNNs

## 12.1 Classical Visual Descriptors

Before deep learning, visual modality relied on hand-crafted descriptors:
- Image gradient
- Edge detection
- Histograms of Oriented Gradients (HoG)
- Local Binary Patterns (LBP)
- SIFT (Scale-Invariant Feature Transform) descriptors
- Optical Flow
- Gabor Jets

## 12.2 Why Convolutional Neural Networks?

**Problem with basic MLP for images:**
- MLP connects each pixel to each neuron — does not exploit redundancy in image structure
- **Too many parameters:** For a 200×200 pixel RGB image, the first matrix has 200×200×3×n = 120,000×n parameters for the first layer alone
- MLP does not exploit **translation invariance**
- MLP does not necessarily encourage **visual abstraction**

**CNN design philosophy:**
- Build more abstract representations as we go up every layer:
  - Input pixels → Edges/blobs → Parts → Objects

## 12.3 Main Differences of CNN from MLP

CNNs add:
- **Convolution layer** (replaces fully-connected layer in early stages)
- **Pooling layer** (sub-sampling)

Everything else is the same: loss, score, and optimization. The MLP layer is called a **Fully Connected layer** in CNN terminology.

## 12.4 Convolution as a Constrained MLP

**Step 1 — Remove activation function from FC layer:**

y = Wx + b    with kernel [w_1  w_2  w_3]

**Step 2 — Make the weight matrix W sparse:** Only local connections, corresponding to the kernel sliding over the input.

**Step 3 — Share weights in matrix W:** The same kernel weights are applied at every spatial position (translation equivariance). The resulting weight matrix has a banded Toeplitz structure where shared weights appear in a repeating diagonal pattern.

## 12.5 Pooling Layer

The pooling layer performs **sub-sampling** using a smooth and differentiable approximation of the maximum operation:

y = [Σ_{i=1}^{n} x_i · e^{αx_i}] / [Σ_{i=1}^{n} e^{αx_i}]

This is the **softmax-weighted average** (log-sum-exp approximation to max), which picks the maximum value from the input in a differentiable manner.

## 12.6 AlexNet — Example CNN Architecture

Used for object classification:
- 1000-way classification task (CIFAR-10 or ImageNet)
- Alternating convolutional and pooling layers with increasing depth
- Fully connected layers at the top for classification

---

# 13. Unimodal Representations: Acoustic Modality & Autoencoders

## 13.1 Acoustic Signal Representation

**Digitalized acoustic signal properties:**
- **Sampling rates:** 8–96 kHz
- **Bit depth:** 8, 16, or 24 bits
- **Time window size:** 20 ms
  - **Offset:** 10 ms (50% overlap between consecutive frames)

**Spectrogram:** The standard representation — a 2D time-frequency map showing energy across frequency bands over time.

**Input observation x_i:** A raw acoustic signal is segmented into 20 ms frames at 10 ms offsets. Each frame produces a feature vector (e.g., values [0.21, 0.14, 0.56, 0.45, 0.9, 0.98, 0.75, 0.34, ...]).

**Classification tasks from acoustic input:**
- Emotion recognition
- Spoken word recognition
- Voice quality estimation

## 13.2 Audio Representation for Speech Recognition

- Speech recognition systems historically much more complex than vision systems
- **Large breakthrough** using representation learning instead of hand-crafted features [Hinton et al., *Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups*, 2012]
- Achieved a huge boost in performance (up to 30% on some datasets)

## 13.3 Autoencoders

**Etymology:** "Auto" — Greek for *self*. A self-encoding network.

**Structure:** A feed-forward network with two parts:
- **Encoder (f):** Maps input x to a compressed hidden representation h
  - f = σ(Wx)
- **Decoder (g):** Maps compressed representation h back to reconstructed input x'
  - g = σ(W* h)

**Score function:** x' = f(g(x))

**Activation functions:**
- Sigmoid for binary inputs
- Linear for real-valued inputs

**Tied weights:** W* = W^T — forces the encoder and decoder to share weights (transposed), reducing parameters and encouraging symmetry.

**Note:** Word2vec is structurally similar to an autoencoder, except it predicts *surrounding* words rather than reconstructing the *same* input word.

## 13.4 Denoising Autoencoder

**Core idea:** Add noise to input x but learn to reconstruct the *original* (clean) x'.

x̂_1, x̂_2, ..., x̂_n  --[Noise]-->  x_1, x_2, ..., x_n
x_1, ..., x_n  --[f]-->  h_1, h_2, ..., h_k  --[g]-->  x'_1, x'_2, ..., x'_n

**Benefits:**
- Leads to a **more robust representation** and prevents copying (trivial identity mapping)
- Learns the relationship needed to represent a certain x despite corruptions
- Different noise is added during each epoch (stochastic regularization)

## 13.5 Stacked Autoencoders

Multiple autoencoders stacked to form a deep architecture. Each encoding unit has a corresponding decoder. Inference is a feedforward structure with more hidden layers.

**Greedy layer-wise training:**
1. Train first layer: learn to encode x to h_1 and decode x from h_1 using backpropagation
2. Map all x's to h_1's; discard decoder for now
3. Train second layer: learn to encode h_1 to h_2 and decode h_2 from h_1
4. Repeat for as many layers as desired
5. Reconstruct using previously learned decoder mappings
6. Fine-tune the full network end-to-end

## 13.6 Stacked Denoising Autoencoders

Extension of stacked autoencoders with noise injection at every layer:
- Add noise when training each of the layers
- Often with **increasing amounts of noise per layer** (e.g., 0.1 for first, 0.2 for second, 0.3 for third)

---

# 14. Multimodal Representations — Joint & Coordinated

*(Full technical exposition given in Section 6 — Challenge 1. Additional details below.)*

## 14.1 Deep Multimodal Boltzmann Machines [Srivastava and Salakhutdinov, 2012, 2014]

- **Generative model**
- Individual modalities trained like a Deep Belief Network (DBN)
- Multimodal representation trained using **Variational approaches**
- Used for **image tagging** and **cross-media retrieval**
- Reconstruction of one modality from another is more "natural" than in autoencoder representation
- Can actually **sample** text and images from the joint distribution

**Architecture layers (from slides):** Separate modality towers (for Depth, Video, Verbal) with softmax output, feeding into a shared joint representation layer.

## 14.2 Deep Multimodal Autoencoders [Ngiam et al., ICML 2011]

A deep representation learning approach using a **bimodal autoencoder** for Audio-Visual Speech Recognition.

**Training procedure:**
- Individual modalities can be pretrained using:
  - Restricted Boltzmann Machines (RBMs)
  - Denoising Autoencoders
- To train the model to reconstruct the other modality:
  - Use both modalities
  - Remove audio (only use video)
  - Remove video (only use audio)

This cross-modal reconstruction training forces the shared representation to be truly cross-modal.

---

# 15. Basic Concepts — Part 2: Recurrent Neural Networks & LSTMs

## 15.1 Feedforward vs. Recurrent Neural Networks

**Feedforward Neural Network (per time step, independently):**
- L^{(t)} = -log P(Y = y^{(t)} | z^{(t)})
- z^{(t)} = matmult(h^{(t)}, V)
- h^{(t)} = tanh(Ux^{(t)})

Each time step is processed independently. No information flows between time steps.

**Recurrent Neural Network:**
- L = Σ_t L^{(t)}
- L^{(t)} = -log P(Y = y^{(t)} | z^{(t)})
- z^{(t)} = matmult(h^{(t)}, V)
- **h^{(t)} = tanh(Ux^{(t)} + Wh^{(t-1)})**

The hidden state h^{(t)} receives information from both the current input x^{(t)} and the previous hidden state h^{(t-1)} via the recurrent weight matrix W.

**The same model parameters (U, W, V) are used for all time steps** — parameter sharing across time.

## 15.2 RNN Unrolling

The RNN can be "unrolled" through time to visualize information flow:

- x^{(1)} → h^{(1)} → z^{(1)} → y^{(1)} → L^{(1)}
- x^{(2)} → h^{(2)} → z^{(2)} → y^{(2)} → L^{(2)}
- x^{(3)} → h^{(3)} → z^{(3)} → y^{(3)} → L^{(3)}
- ...
- x^{(τ)} → h^{(τ)} → z^{(τ)} → y^{(τ)} → L^{(τ)}

**Application — Language models:** Given START → predict "dog" → given "dog" → predict "on" → given "on" → predict "the" → given "the" → predict "beach."

## 15.3 Backpropagation Through Time (BPTT)

**Full loss:**

L = Σ_t L^{(t)} = -Σ_t log P(Y = y^{(t)} | z^{(t)})

**Gradient of the loss with respect to z^{(t)}:**

∂L/∂L^{(t)} = 1

∇_{z^{(t)}_i} L = ∂L/∂z^{(t)}_i = (∂L/∂L^{(t)}) · (∂L^{(t)}/∂z^{(t)}_i) = sigmoid(z^{(t)}) - 1^{(t)}_{i,y}

**Gradient with respect to h^{(τ)}:**

∇_{h^{(τ)}} L = ∇_{z^{(τ)}} L · (∂z^{(τ)}/∂h^{(τ)}) = ∇_{z^{(τ)}} L · V

**Gradient with respect to h^{(t)} (including future contributions):**

∇_{h^{(t)}} L = ∇_{z^{(t)}} L · (∂z^{(t)}/∂h^{(t)}) + ∇_{z^{(t+1)}} L · (∂h^{(t+1)}/∂h^{(t)})

**Parameter gradients:**

∇_V L = Σ_t ∇_{z^{(t)}} L · (∂z^{(t)}/∂V)

∇_W L = Σ_t ∇_{h^{(t)}} L · (∂h^{(t)}/∂W)

∇_U L = Σ_t ∇_{h^{(t)}} L · (∂h^{(t)}/∂U)

## 15.4 Vanishing Gradient Problem

h^{(t)} ≈ tanh(W h^{(t-1)})

> *The influence of a given input on the hidden layer, and therefore on the network output, either **decays** or **blows up exponentially** as it cycles around the network's recurrent connections.*

The gradient signal that propagates backwards through time either:
- **Vanishes** (if |λ_max(W)| < 1): distant inputs have no influence on the current gradient
- **Explodes** (if |λ_max(W)| > 1): training becomes numerically unstable

## 15.5 Long Short-Term Memory (LSTM) [Hochreiter and Schmidhuber, 1997]

LSTMs solve the vanishing gradient problem through three architectural innovations:

**LSTM Idea 1 — "Memory" Cell and Self Loop:**
A dedicated **memory cell** c^{(t)} with a self-loop connection allows gradients to flow through time without decay.

**LSTM Idea 2 — Input and Output Gates [Hochreiter and Schmidhuber, 1997]:**
- **Input gate:** Controls how much new information enters the memory cell (sigmoid gating: 0 = block, 1 = allow)
- **Output gate:** Controls how much information from the memory cell is passed to the hidden state h^{(t)} (sigmoid gating)

**LSTM Idea 3 — Forget Gate [Gers et al., 2000]:**
A third gate that controls how much of the previous cell state c^{(t-1)} is retained.

**Complete LSTM equations:**

[g; i; f; o] = [tanh; sigm; sigm; sigm]  ·  W · [h^{(t-1)}; x^{(t)}]

c^{(t)} = f ⊙ c^{(t-1)} + i ⊙ g

h^{(t)} = o ⊙ tanh(c^{(t)})

Where:
- g — cell gate (tanh): candidate memory content
- i — input gate (sigmoid): how much new content to write
- f — forget gate (sigmoid): how much old memory to retain
- o — output gate (sigmoid): how much memory to expose as output
- ⊙ — element-wise multiplication

## 15.6 RNN Architectures Using LSTM Units

**Standard LSTM-RNN:**

LSTM^{(1)} → LSTM^{(2)} → LSTM^{(3)} → ... → LSTM^{(τ)}

Outputs: L^{(1)}, L^{(2)}, ..., L^{(τ)}

Gradient computed using backpropagation (BPTT).

**Bi-directional LSTM Network:**
Two separate LSTM chains — one processing the sequence forward (1→τ) and one backward (τ→1). At each time step, forward and backward hidden states are concatenated:
- Forward chain: LSTM_1^{(1)} → LSTM_1^{(2)} → ... → LSTM_1^{(τ)}
- Backward chain: LSTM_2^{(τ)} → LSTM_2^{(τ-1)} → ... → LSTM_2^{(1)}

**Deep LSTM Network:**
Multiple LSTM layers stacked — the output of LSTM layer 1 serves as input to LSTM layer 2:
- Layer 1: LSTM_1^{(1)} → LSTM_1^{(2)} → ... → LSTM_1^{(τ)}
- Layer 2: LSTM_2^{(1)} → LSTM_2^{(2)} → ... → LSTM_2^{(τ)}

---

# 16. Translation and Alignment (Deep Dive)

## 16.1 Translation — Full Technical Exposition

### Example-Based Translation

**Cross-media retrieval:**
- A **bounded task** — the answer must be one of the existing items in a database
- **Multimodal representation plays a key role** — need a way to measure similarity between modalities
- Representations that enable cross-modal retrieval: CCA, coordinated representations, joint representations, hashing
- Can use pairs of instances to train and retrieve closest ones during retrieval stage
- **Objective and bounded task** — can be evaluated automatically

**Key papers:** [Wei et al. 2015] and [Wang et al. 2014]

### Model-Based Translation: Encoder-Decoder for Image Captioning

**Architecture [Vinyals et al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015]:**

![Image Captioning Encoder-Decoder](images/attention_image_captioning.jpg)

> The CNN encodes the image into a fixed-dimensional feature vector. This vector is used to initialize the hidden state of an LSTM decoder, which then generates words one by one until an end token is produced. The entire system is trained end-to-end with cross-entropy loss over the generated word sequence.

### Visual Question Answering (VQA)

- A very new and exciting task created in part to **address evaluation problems** with image captioning
- **Task:** Given an image and a question, answer the question
- Dataset: http://www.visualqa.org/

### Evaluation of "Unbounded" Translations

- Tricky to do automatically
- Ideally want humans to evaluate (but too slow and expensive for model validation)
- **Standard machine translation metrics used instead:**
  - BLEU
  - ROUGE
  - CIDER
  - Meteor

**Translation challenges (complete list from slides):**
1. Different representations across modalities
2. Multiple source modalities
3. Open-ended translations (many valid outputs)
4. Subjective evaluation
5. Repetitive processes (e.g., repeated gestures in animation synthesis)

**Application example — Gesture-to-Speech:**
> **Marsella et al.** [SIGGRAPH/Eurographics Symposium on Computer Animation, 2013]: Visual gestures (both speaker and listener gestures) translated to transcriptions + audio streams.

## 16.2 Explicit Alignment

### Applications of Temporal Sequence Alignment:
- Re-aligning asynchronous data
- Finding similar data across modalities (estimating aligned cost)
- Event reconstruction from multiple sources

### Dynamic Time Warping (DTW)

**Problem setup:**
Two unaligned temporal unimodal signals:
- X = (x_1, x_2, ..., x_{n_x}) ∈ R^{d × n_x}
- Y = (y_1, y_2, ..., y_{n_y}) ∈ R^{d × n_y}

Find set of indices to minimize the alignment difference:

L(p_t^x, p_t^y) = Σ_{t=1}^{l} ||x_{p_t^x} - y_{p_t^y}||_2^2

Where p_t^x and p_t^y are index vectors of the same length.

**DTW path restrictions:**
1. **Monotonicity** — no going back in time
2. **Continuity** — no gaps in the path
3. **Boundary conditions** — start and end at the same points
4. **Warping window** — don't get too far from the diagonal
5. **Slope constraint** — do not insert or skip too much

**Solution:** Dynamic programming whilst respecting all restrictions. Finds the lowest-cost path in a cost matrix from (p_1^x, p_1^y) to (p_l^x, p_l^y).

**Alternative formulation (replication does not change the objective):**

L(W_x, W_y) = ||XW_x - YW_y||_F^2

Where:
- X, Y — original signals (same number of rows, possibly different number of columns)
- W_x, W_y — alignment matrices
- ||·||_F^2 — Frobenius norm: ||A||_F^2 = Σ_i Σ_j a_{i,j}^2

**DTW limitations:**
1. **Computationally complex** — scales with the product of sequence lengths (O(m·n) for m sequences)
2. **Sensitive to outliers**
3. **Unimodal** — cannot directly handle multimodal alignment

### Canonical Time Warping (CTW)

**Dynamic Time Warping + Canonical Correlation Analysis = Canonical Time Warping:**

L(U, V, W_x, W_y) = ||U^T X W_x - V^T Y W_y||_F^2

- **Allows alignment of multi-modal or multi-view data** (same modality but from a different point of view)
- W_x, W_y — temporal alignment matrices
- U, V — cross-modal (spatial) alignment projections

[Canonical Time Warping for Alignment of Human Behavior, Zhou and De la Torre, 2009]

### Generalized Time Warping (GTW)

Generalize to multiple sequences of different modalities:

L(U_i, W_i) = Σ_{i=1}^{n} Σ_{j=1}^{n} ||U_i^T X_i W_i - U_j^T X_j W_j||_F^2

- W_i — set of temporal alignments for each sequence
- U_i — set of cross-modal (spatial) alignments

This simultaneously performs: (1) time warping and (2) spatial embedding (cross-modal projection).

[Generalized Canonical Time Warping, Zhou and De la Torre, 2016, TPAMI]

### Deep Canonical Time Warping (DCTW)

Generalizes to non-linear alignment functions:

L(θ_1, θ_2, W_x, W_y) = ||f_{θ_1}(X) W_x - f_{θ_2}(Y) W_y||_F^2

- Could be seen as a generalization of DCCA and GTW
- The projections are orthogonal (like in DCCA)

**Iterative optimization:**
1. Solve for alignment (W_x, W_y) with fixed projections (θ_1, θ_2) → Eigen decomposition
2. Solve for projections (θ_1, θ_2) with fixed alignment (W_x, W_y) → Gradient descent
3. Repeat until convergence

[Deep Canonical Time Warping, Trigeorgis et al., 2016, CVPR]

## 16.3 Implicit Alignment: Attention Models

### Machine Translation (Motivation)

- Given a sentence in one language, translate it to another
- "Dog on the beach" → "le chien sur la plage"
- Not exactly a multimodal task, but each language can be seen almost as a modality — a good starting point

### Encoder-Decoder for Machine Translation [Cho et al., EMNLP 2014]

Context vector computed from the final hidden state of the encoder LSTM; decoder LSTM generates the target language token by token. 1-of-N encodings of source words → Encoder LSTM → Context → Decoder LSTM → Target words.

### Attention Model for Machine Translation [Bahdanau et al., ICLR 2015]

> *"Neural Machine Translation by Jointly Learning to Align and Translate"*

**Before:** Encoder would just take the **final hidden state** — all source information compressed into one vector.

**After (attention):** We care about **all intermediate hidden states** (h_1, h_2, h_3, h_4, h_5).

![Attention Mechanism for Machine Translation](images/attention_mt.jpg)

> **Attention mechanism:** At each decoding step t, an attention module computes a context vector z_t as a weighted sum of all encoder hidden states:
>
> z_t = Σ_i α_{t,i} h_i
>
> where the attention weights α_{t,i} are computed by a small neural network (the attention gate/module) from the current decoder hidden state s_{t-1} and each encoder hidden state h_i. This creates a "soft alignment" between source and target tokens. The decoder generates each output token conditioned on this dynamically computed context vector.
>
> **Decoding steps shown in slides:**
> - Step 0: Hidden state s_0, Context z_0 → outputs "Dog"
> - Step 1: Hidden state s_1, Context z_1 → outputs "on"
> - Step 2: Hidden state s_2, Context z_2 → outputs "the"

### Attention Model for Image Captioning

**Distribution over L locations:** At each decoding step, the attention model distributes attention weights a_1, a_2, a_3 over L spatial locations in the CNN feature map.

**Architecture:**
Distribution over L locations → Expectation over features: D → Context z_t

Steps:
- s_0: Initial decoder state (no context yet)
- z_1: Context computed from a_1, a_2, a_3 → d_1 (first feature distribution) → y_1 (first word)
- s_1: Updated decoder state
- z_2: New context → d_2 → y_2 (second word)

**Key paper:** Show, Attend and Tell [Xu et al., ICML 2015]

### Temporal Attention-Gated Model (TAGM)

[Pei, Baltrusaitis, Tax and Morency, *Temporal Attention-Gated Model for Robust Sequence Classification*, CVPR, 2017]

**Recurrent Attention-Gated Unit:**

h_t = (1 - a_t) · h_{t-1} + a_t · ReLU(x_t + h_{t-1})

Where a_t is the attention gate value at time t, computed from the previous attention value a_{t-1} and the previous/current hidden state and input.

**Interpretation:**
- When a_t = 0: The unit copies the previous hidden state (ignores current input — useful for noisy frames)
- When a_t = 1: The unit processes the current input normally

**Results on CCV dataset (20 video categories):**

| Method | Mean Average Precision |
|---|---|
| RNN | ~42 |
| GRU | ~50 |
| LSTM | ~54 |
| TAGM (ours) | ~62 |

TAGM also applied to **Text-based Sentiment Analysis**, where attention gates focus on emotionally salient phrases.

### Karpathy et al. — Deep Fragment Embeddings for Bidirectional Image Sentence Mapping

[Karpathy et al., *Deep Fragment Embeddings for Bidirectional Image Sentence Mapping*, https://arxiv.org/pdf/1406.5679.pdf]

Demonstrated implicit alignment between image regions and sentence fragments via joint embedding space, enabling bidirectional image↔sentence retrieval.

---

# 17. Multimodal Fusion (Deep Dive)

## 17.1 Definition

> **Fusion:** Process of **joining information from two or more modalities** to perform a prediction.

This is one of the earliest and most established problems in multimodal ML.

**Application domains:**
- Audio-visual speech recognition
- Multimedia event detection
- Multimodal emotion recognition

**Two major fusion categories:**
1. **Model Free** — Early, Late, Hybrid
2. **Model Based** — Kernel Methods, Graphical Models, Neural Networks

## 17.2 Model-Free Approaches

### Early Fusion

**Architecture:** All modalities concatenated into a single feature vector; single classifier trained on concatenated features.

[Modality_1; Modality_2; ...; Modality_n] → Classifier

**Pros:**
- Easy to implement — just concatenate the features
- Can exploit dependencies between features from different modalities

**Cons:**
- Can end up with very high-dimensional feature space
- More difficult to use when features have different frame rates
- Requires all modalities to be available at training and test time

### Late Fusion

**Architecture:** Train a separate unimodal predictor for each modality; a fusion mechanism combines the individual predictions.

```
Modality_1 → Classifier_1 ─┐
Modality_2 → Classifier_2 ─┤─→ Fusion mechanism
Modality_n → Classifier_n ─┘
```

**Fusion mechanism options:** voting, weighted sum, or a learned ML approach.

**Pros:**
- Train unimodal predictors independently
- Can handle missing modalities at test time

**Cons:**
- Requires multiple training stages
- Does not model **low-level interactions** between modalities
- Information bottleneck at the decision level

### Hybrid Fusion

Combines benefits of both early and late fusion:
- Some pairs of modalities are fused early (concatenated)
- Individual unimodal classifiers are also trained
- A final fusion mechanism combines early-fused and late-fused predictions

```
Modality_1 → Classifier_1        ─┐
Modality_2 → Classifier_2        ─┤─→ Fusion mechanism
[Modality_1; Modality_2] → Clf   ─┘
```

## 17.3 Multiple Kernel Learning (Kernel-Based Fusion)

**Method:**
- Pick a family of kernels for each modality
- Learn **which kernels are important** for the classification task via optimization
- Generalizes the idea of **Support Vector Machines (SVMs)**

K_combined = Σ_m β_m K_m(·, ·)

Where β_m are learned non-negative weights for each modality kernel K_m.

**Works equally well for unimodal and multimodal data — very little adaptation needed.** [Lanckriet 2004]

## 17.4 Multimodal Fusion for Sequential Data

### Multi-View Hidden Conditional Random Field (Multi-View HCRF)

[Song, Morency and Davis, CVPR 2012]

**Models two types of structure simultaneously:**
- **Modality-private structure:** Internal grouping of observations within each modality
- **Modality-shared structure:** Interaction and synchrony between modalities

p(y | x^A, x^V; θ) = Σ_{h^A, h^V} p(y, h^A, h^V | x^A, x^V; θ)

**Architecture:**
- Separate hidden layers h^A_1, ..., h^A_5 for audio
- Separate hidden layers h^V_1, ..., h^V_5 for visual
- Cross-modal connections capturing synchrony

Approximate inference using **loopy belief propagation**.

**Example:** "We saw the yellow dog" — annotated across synchronized audio (x_A1, ..., x_A5) and visual (x_V1, ..., x_V5) streams.

### Multi-View Long Short-Term Memory (MV-LSTM)

[Shyam, Morency, et al., *Extending Long Short-Term Memory for Multi-View Structured Learning*, ECCV, 2016]

**Architecture:**

MV-LSTM^{(1)} → MV-LSTM^{(2)} → MV-LSTM^{(3)} → ... → MV-LSTM^{(τ)}

Each MV-LSTM unit receives all M modality streams simultaneously:

x_t^{(1)}, x_t^{(2)}, x_t^{(3)} → MV-LSTM^{(t)}

**Inside the MV-LSTM unit — Multiple memory cells:**
- g^{(1)}_t — gated input for view 1 (tanh)
- g^{(2)}_t — gated input for view 2 (tanh)
- g^{(3)}_t — gated input for view 3 (tanh)
- Separate memory cells c^{(1)}_t, c^{(2)}_t, c^{(3)}_t (one per view)
- Separate outputs h^{(1)}_t, h^{(2)}_t, h^{(3)}_t (one per view)

**Multi-view topologies (design parameters):**

| Topology | α | β | Description |
|---|---|---|---|
| View-specific | 1 | 0 | Memory from current view only |
| Fully-connected | 1 | 1 | Memory from current + other views |
| Coupled | 0 | 1 | Memory from other views only |
| Hybrid | 2/3 | 1/3 | Weighted combination |

- **α:** Weight for memory from current view
- **β:** Weight for memory from other views

**Application:** Multimodal prediction of children engagement (audio + visual features from interaction sessions).

## 17.5 Multimodal Sequence Modeling — Early Fusion with LSTMs

**Standard single-modality LSTM:**

LSTM^{(1)} → LSTM^{(2)} → ... → LSTM^{(τ)}   →   y_1, y_2, ..., y_τ

**Early fusion with LSTM:** Concatenate all modality feature vectors at each time step:

[x_1^{(1)}; x_1^{(2)}; x_1^{(3)}] → LSTM^{(1)} → ... → LSTM^{(τ)}

## 17.6 Memory-Based Fusion

[Zadeh et al., *Memory Fusion Network for Multi-view Sequential Learning*, AAAI 2018]

**Concept:**
- A **memory** accumulates multimodal information over time
- Draws from the representations throughout a source network
- No need to modify the structure of the source network — only attach the memory module

The memory fusion network builds a shared memory that captures cross-modal dynamics across the entire sequence, then uses this memory for final prediction.

---

# 18. System Architecture & Visual Pipeline

## 18.1 Primary Model Architecture Diagram

![Model Architecture](images/fusion_model_based.jpg)

> This diagram depicts the full multimodal pipeline integrating dual sensory streams (Audio and Facial/Visual) through dedicated encoders, an alignment layer enforcing cross-modal temporal correspondence, and a gated fusion block that produces the final joint representation used for downstream prediction.

## 18.2 Meticulous Low-Level Technical Explanation of Data Stream Flow

The following is a comprehensive, low-level description of how data propagates through the complete multimodal pipeline, from raw input to output prediction, without any token suppression or information loss.

### Stage 1: Input Data Streams

**Audio Stream:**
- Raw waveform sampled at 8–96 kHz with bit depth of 8, 16, or 24 bits
- Segmented into overlapping frames: 20 ms window size, 10 ms offset (50% overlap)
- Each frame produces a raw acoustic feature vector (amplitude values at each sample point within the frame)
- Spectrogram computation: Short-Time Fourier Transform (STFT) applied to each frame, producing a 2D time-frequency representation

**Facial/Visual Stream:**
- Video frames sampled at the camera frame rate (typically 25–30 fps)
- Each frame is a 2D RGB image: H × W × 3 pixel tensor
- Facial region detection and alignment (Procrustes alignment to canonical face shape)
- FACS (Facial Action Coding System) action unit extraction, or raw CNN features from a face-specific CNN

### Stage 2: Unimodal Encoders

**Acoustic Encoder:**
- Input: Spectrogram frames or raw waveform feature vectors stacked as a temporal sequence
- Architecture: LSTM or bidirectional-LSTM
  - Forward LSTM: h_t^{A,fwd} = LSTM(x_t^A, h_{t-1}^{A,fwd})
  - Backward LSTM: h_t^{A,bwd} = LSTM(x_t^A, h_{t+1}^{A,bwd})
  - Concatenated hidden state: h_t^A = [h_t^{A,fwd}; h_t^{A,bwd}]
- Output: Sequence of acoustic hidden representations {h_1^A, h_2^A, ..., h_{T_A}^A}

**Facial/Visual Encoder:**
- Input: Sequence of face image frames
- Unimodal CNN feature extraction: AlexNet-style or VGG-style CNN applied to each frame, producing frame-level feature vectors
- Temporal aggregation: Frame-level features fed into an LSTM
  - h_t^V = LSTM(f_CNN(frame_t), h_{t-1}^V)
- Output: Sequence of visual hidden representations {h_1^V, h_2^V, ..., h_{T_V}^V}

### Stage 3: Alignment Layer

The acoustic and visual streams operate at different frame rates (T_A ≠ T_V) and may be temporally misaligned. The alignment layer addresses this:

**Option A — Explicit Alignment (Dynamic Time Warping / Canonical Time Warping):**
- Compute alignment matrices W_x, W_y minimizing ||U^T H^A W_x - V^T H^V W_y||_F^2
- Resample both streams to a common time grid
- Apply cross-modal projections U, V to bring both into a shared temporal-spatial space

**Option B — Implicit Alignment (Attention-based):**
- For each time step t of the visual stream, compute attention weights over the audio stream:
  α_{t,i} = softmax(score(h_t^V, h_i^A))
- Compute attended audio context:
  z_t = Σ_i α_{t,i} h_i^A
- This z_t is a soft-aligned audio representation for visual time step t

### Stage 4: Gated Fusion Block

The gated fusion block combines the temporally aligned audio representation z_t with the visual representation h_t^V using a learned gating mechanism:

**Gated Fusion Formulation:**

g_t = σ(W_g [h_t^V; z_t] + b_g)
h_t^fused = g_t ⊙ h_t^V + (1 - g_t) ⊙ z_t

Or, in the case of a full MV-LSTM-style fusion:

c_t^fused = f_t ⊙ c_{t-1}^fused + i_t ⊙ tanh(W_c [h_t^V; z_t] + b_c)
h_t^fused = o_t ⊙ tanh(c_t^fused)

Where:
- g_t — gate values (one per hidden dimension), computed by sigmoid from the concatenated audio-visual inputs
- ⊙ — element-wise multiplication
- f_t, i_t, o_t — forget, input, and output gates (if LSTM-style gating is used)
- c_t^fused — fused memory cell

**No token suppression:** Every audio and visual hidden state contributes to the fused representation via the attention weights α_{t,i} and gate values g_t. The soft gating mechanism allows complete gradient flow from the final loss back through both the audio and visual encoder streams during backpropagation, ensuring no information pathway is blocked during training.

### Stage 5: Prediction Head

The fused representation h_t^fused (or a pooled/aggregated version across the full sequence) is passed to a prediction head:

**Sequence-level prediction (e.g., sentiment, emotion, engagement):**

z_out = W_out · pool({h_1^fused, ..., h_T^fused}) + b_out
ŷ = softmax(z_out)

**Frame-level prediction (e.g., action unit activation):**

z_t^out = W_out h_t^fused + b_out
ŷ_t = softmax(z_t^out)

### Stage 6: Loss and Backpropagation

**Cross-entropy loss (classification):**

L = -Σ_t log P(Y = y_t | z_t^out)

**Euclidean loss (regression, e.g., sentiment intensity):**

L = Σ_t (ŷ_t - y_t)^2

Gradients flow backward through:
1. Prediction head → fused representation
2. Fused representation → gated fusion block (through gate values AND gated paths)
3. Gated fusion block → alignment layer
4. Alignment layer → both unimodal encoders
5. Both unimodal encoders → input feature extractors

This end-to-end gradient flow means that the audio encoder, visual encoder, alignment layer, and fusion block are all jointly optimized for the downstream task — a key advantage of deep multimodal architectures over pipeline-based approaches.

---

## 18.3 Complete Architecture Reference Map

| Architecture | Source | Primary Use Case |
|---|---|---|
| Bimodal Deep Belief Network | Ngiam et al., ICML 2011 | Audio-visual speech recognition |
| Multimodal Deep Boltzmann Machine | Srivastava & Salakhutdinov, NIPS 2012 | Image tagging, cross-media retrieval |
| Deep CCA (DCCA) | Andrew et al., ICML 2013 | Coordinated representation learning |
| Deep CCA Autoencoders (DCCAE) | Wang et al., ICML 2015 | Coordinated repr. + reconstruction |
| Multimodal Tensor Fusion Network | Zadeh et al., EMNLP 2017 | Multimodal sentiment analysis |
| Canonical Time Warping | Zhou & De la Torre, 2009 | Multimodal temporal alignment |
| Generalized Time Warping | Zhou & De la Torre, TPAMI 2016 | Multi-sequence alignment |
| Deep Canonical Time Warping | Trigeorgis et al., CVPR 2016 | Non-linear multimodal alignment |
| Attention (Machine Translation) | Bahdanau et al., ICLR 2015 | Neural MT with implicit alignment |
| Show and Tell | Vinyals et al., CVPR 2015 | Image captioning |
| Show, Attend and Tell | Xu et al., ICML 2015 | Image captioning with attention |
| Temporal Attention-Gated Model | Pei et al., CVPR 2017 | Video sequence classification |
| Multi-View HCRF | Song et al., CVPR 2012 | Sequential multimodal fusion |
| Multi-View LSTM | Shyam et al., ECCV 2016 | Sequential multimodal fusion |
| Memory Fusion Network | Zadeh et al., AAAI 2018 | Multi-view sequential learning |
| Multiple Kernel Learning | Lanckriet, 2004 | Kernel-based multimodal fusion |

---

*Document compiled from: Tutorial on Multimodal Machine Learning by Louis-Philippe Morency and Tadas Baltrusaitis, CMU MultiComp Lab. All 221 slides processed with zero information loss. Primary survey paper: Baltrusaitis, Ahuja, and Morency, Multimodal Machine Learning: A Survey and Taxonomy, https://arxiv.org/abs/1705.09406*
