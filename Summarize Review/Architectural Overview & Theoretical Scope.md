# Project Foundations: Architectural Overview & Theoretical Scope

> **Source Document:** *Vision + X: A Survey on Multimodal Learning in the Light of Data*
> Ye Zhu (Princeton University) · Yu Wu (Wuhan University) · Nicu Sebe (University of Trento) · Yan Yan (Illinois Institute of Technology)
> Published: *Journal of LaTeX Class Files*, 2022 · arXiv:2210.02884v2 [cs.CV] · Revised: 7 June 2024

---

## Executive Preamble

Human beings perceive and communicate with the world through a **multisensory system**: seeing objects, hearing sounds, speaking languages, writing and reading texts. The information from these varied sources is processed by **different, anatomically distinct parts of the human brain**. The occipital lobe acts as the primary center for visual processing, interpreting the distance and locations of objects. The temporal lobe processes auditory information, helping us understand sounds. Language comprehension, facilitated by Wernicke's area in the posterior superior temporal lobe, is crucial for decoding both written and spoken words. Other sensory information — including touch and movement — is processed by distinct brain areas. These integrated yet distinct functions form a complex, harmonious human sensing system.

The **specialized divisions in human neural processing**, which highlight both unique and shared characteristics across different modalities, directly inspire the framing of multimodal machine learning in the light of data. Historically, vision, audio, and textual data were studied in **separate research fields** — computer vision, digital signal processing, and natural language processing, respectively. The ultimate objective of bringing **true intelligence to machines** has driven modern AI beyond single-perception-perspective exploitation into an era where the interplay of multiple sensing systems is studied in a **collaborative, brain-inspired manner**.

---

## Complete Survey Taxonomy & Structural Overview

![Overall Survey Structure](path/to/fig1_survey_structure.jpg)

The figure above encodes the full hierarchical taxonomy of the survey. Data modalities branch left (Vision → Images/Videos; Audio → Music/Speech/Ambient Sound; Text → Captions/Dialog/Q&A/ASR&OCR; Others → Graph/Optical Flow/Point Clouds & Meshes/Human-Centric Motions). Representation Learning branches center (Network Backbones: CNNs/RNNs/Transformers; Supervised; Non-supervised). Applications branch right into Discriminative Applications (Vision+Audio, Vision+Text, Vision+Audio+Text) and Generative Applications (Vision+Audio, Vision+Text), each with their own generative backbone families (VAEs/GANs/DPMs).

The paper's structural sequence is as follows:

- **Section 2 — Data Analysis:** Intrinsic nature of modalities (vision, audio, text, others), characteristics, commonalities, and a comprehensive dataset catalogue.
- **Section 3 — Multimodal Representation Learning:** Network architectures and evaluation; supervised learning setting; non-supervised (unsupervised/weakly-supervised/self-supervised) setting.
- **Section 4 — Discriminative Applications:** Vision+Audio, Vision+Text, Vision+Audio+Text task families.
- **Section 5 — Generative Applications:** Generative backbone models; Vision+Audio generation; Vision+Text generation.
- **Section 6 — Further Discussion:** Insights from data–methodology design connections; future directions and challenges.
- **Section 7 — Conclusions.**
- **Appendix A — Multimodal Datasets:** Full catalogue with data scale, annotations, applicable tasks.
- **Appendix B — Evaluation for Synthesized Data:** Metrics for visual, audio, and textual synthesized data.

---

## 🧠 Theoretical Framework: Modality vs. Medium

### Definition of Multimodal

The term **multimodal** in the context of this survey refers to the simultaneous involvement of **multiple data modalities** — distinct information sources and representation formats — that each carry unique statistical characteristics, semantic content, and perceptual properties. These modalities are not interchangeable; they occupy different feature spaces and require different architectural processing pipelines.

From a probabilistic standpoint, a **multimodal distribution** is one that exhibits **multiple modes** — i.e., distinct peaks or local maxima in the Probability Density Function (PDF). In the context of multimodal machine learning, this statistical intuition extends to the idea that data drawn from heterogeneous sources (vision, audio, text, motion) occupies **distinct regions of a joint probability distribution**, each with its own marginal density structure, correlation properties, and information-theoretic content. The challenge of multimodal learning is therefore to learn a **joint or aligned representation** that captures both the shared structure (cross-modal correspondences) and the unique structure (within-modality idiosyncrasies) of these distinct distributional modes.

### Modality

A **modality** refers to:

1. A **sensory modality** — a channel through which a human or machine perceives information from the world. Classical sensory modalities include vision, audition, touch, smell, and taste.
2. A **representation format** — the particular data structure or encoding in which information from that channel is captured and stored (e.g., pixel arrays for images, waveforms or spectrograms for audio, token sequences for text, 3D coordinates for point clouds).
3. A **channel of communication** — the specific type of signal carrier that conveys meaning in a multimodal system. Different channels have different bandwidths, noise characteristics, temporal resolutions, spatial resolutions, and semantic densities.

Within this survey, the principal modalities studied are **vision** (images and videos), **audio** (music, speech, ambient sound), **text** (captions, dialog, question-and-answer pairs, ASR/OCR text), and **others** (graphs, optical flow, point clouds and meshes, human-centric motions).

### Medium

A **medium** is the **means of communication or transmission** — the physical or digital delivery infrastructure through which a modality is conveyed. The medium is distinct from the modality: for example, the medium of a radio broadcast (electromagnetic waves) carries the audio modality (speech), while the medium of a printed book (paper and ink) carries the textual modality. In machine learning systems, the medium is typically the digital storage format and transport protocol (JPEG files, WAV files, UTF-8 encoded strings, HDF5 arrays), distinct from the perceptual modality itself.

### Historical Evolution: Multimodal vs. Multimedia

The paradigm shift from **multimedia** to **multimodal** research represents a fundamental change in scientific objective:

- **Multimedia** (historical framing): Concerned primarily with the **storage, transmission, retrieval, and rendering** of heterogeneous media types — audio, video, text — as discrete content artifacts. The emphasis was on engineering infrastructure: codecs, streaming protocols, database indexing, user interfaces. Multimedia systems juxtapose different media but do not necessarily model the **semantic and statistical relationships** among them.

- **Multimodal Machine Learning** (contemporary framing): Concerned with **jointly modeling, representing, and reasoning about** information from multiple sensory modalities simultaneously. The emphasis is on learning algorithms that exploit **cross-modal correspondences**, **complementarity**, and **redundancy** to achieve tasks that neither modality could accomplish alone. The biological inspiration is the human brain's ability to fuse information from multiple senses into a unified perceptual experience.

The transition reflects a move from **content management** (multimedia) to **content understanding and generation** (multimodal ML), enabled by the deep learning revolution, large-scale paired datasets, and transformer architectures capable of processing heterogeneous sequential data.

---

# Section 2: Data Analysis — Intrinsic Modality Characteristics

## 2.1 Vision

Visual data is categorized into **images** and **videos**. As a primary information source in both human sensory systems and computer vision literature, visual data is often considered **"raw data"** — it is of high dimensionality, encompasses a wealth of features and details, and represents rich visual content. However, the redundancy in continuous spatial (for images) and spatio-temporal (for videos) aspects poses significant challenges for processing, analysis, and efficient utilization in multimodal learning tasks.

### 2.1.1 Images

![Image Feature Descriptors and CNN Pipeline](path/to/image_feature_pipeline.jpg)

Images are fundamental to computer vision research, characterized by their **inherent invariance to transformations**. This key attribute — that the semantic content of an image remains stable across rotations, translations, and scale changes — drives the development of classic image processing methods and deep learning techniques such as CNNs to extract meaningful visual features.

**Pre-Deep Learning Era:**
Image processing and computer vision research primarily aimed at deciphering image content and patterns through a **manual feature extraction and analysis pipeline** using machine learning techniques. Three prominent examples of handcrafted image feature descriptors:

- **SIFT (Scale-Invariant Feature Transform)** [Lowe, 2004]: Detects and describes local features invariant to scale and rotation. Widely used for keypoint matching and object recognition.
- **HOG (Histogram of Oriented Gradients)** [Dalal & Triggs, 2005]: Encodes the distribution of gradient orientations in localized image regions. Particularly effective for human detection tasks.
- **SURF (Speeded-Up Robust Feature)** [Bay et al., 2006]: A faster approximation to SIFT using integral images and Haar wavelets, enabling real-time feature detection and description.

After feature extraction, machine learning algorithms such as **Support Vector Machines (SVM)** [Cortes & Vapnik, 1995] and **Principal Component Analysis (PCA)** [Wold et al., 1987] were applied to further analyze the feature data for classification and dimensionality reduction tasks, respectively.

**Deep Learning Era:**
With the rapid development in deep neural network architectures [He et al. 2016 (ResNet); Krizhevsky et al. 2012 (AlexNet); Simonyan & Zisserman 2014 (VGG)] and the availability of large-scale image datasets such as **ImageNet** [Deng et al. 2009; Russakovsky et al. 2015], computer vision entered a new era where the classic procedure of feature extraction and analysis has been **automatically integrated into neural network designs** through end-to-end training with gradient descent.

**Applications extended from:**
- Simple image classification [Simonyan & Zisserman 2014; He et al. 2016]
- To object detection [Zhao et al. 2019]
- To segmentation [Long et al. 2015]

In addition to these **discriminative task applications** (mining patterns from existing images), there is a parallel branch of **generative applications** that synthesize image data using generative neural networks (VAEs, GANs, DPMs — detailed in Section 5).

### 2.1.2 Videos

![I3D Architecture: 3D Convolutional Extension for Video](path/to/i3d_architecture.jpg)

Video is a form of common visual data extensively studied in the computer vision community [Oh et al. 2011; Perazzi et al. 2016; Xu et al. 2016]. Unlike static images, videos **encapsulate information across the temporal dimension**. Human actions in videos are defined by a series of specific movements depicted in consecutive video frames over time — consistency and transformation in the visual context that can only be presented in the format of videos.

This **temporal characteristic** of video data also influences video-based applications, which often require additional understanding and analysis of temporal elements (e.g., actions, motions, optical flow) [Cheng et al. 2017; Sun et al. 2018]. While conventional image representation encoded by neural networks can be applied to individual frames, **extracting video representations necessitates addressing the connections between temporally related frames**.

**Classic Approach:** Extending conventional 2D CNNs into **3D architectures** with an additional temporal dimension. The **I3D model** [Carreira & Zisserman, 2017] is a notable example proposed for action recognition in videos — it inflates 2D convolutional filters into 3D, effectively capturing spatiotemporal patterns jointly rather than treating frames independently.

**Video-Based Application Tasks:**
- **Discriminative:** Video classification (action recognition) [Carreira & Zisserman 2017]; segmentation [Tsai et al. 2016]
- **Generative:** Direct video synthesis [Tulyakov et al. 2018 (MoCoGAN)]
- **Large-Scale Video Generation:** **Sora** from OpenAI [Brooks et al. 2024] stands out as a most recent large-scale video generator, representing the frontier of text-to-video generation.

---

## 2.2 Audio

![Audio Data Representations: Waveform, Mel-Spectrogram, Piano-Roll, MIDI](path/to/fig2_audio_representations.jpg)

The figure shows, from top to bottom: (a) raw audio data in waveform — a 2D signal depicting sound pressure variation measured by air vibration in the time domain; (b) audio data in mel-spectrogram — reflecting frequency content over time; (c) music piece in 1D piano-roll [Dong et al. 2018], where the horizontal axis represents timestamps and the vertical axis represents acoustic pitch; (d) music piece in MIDI [Picazo-Vela & Hernandez 2019], where colors represent different instrument types.

Traditionally, the study of audio processing resided predominantly within the research field of **digital signal processing**. This survey focuses on three primary types of audio data: **music**, **speech**, and **ambient sound**. Each holds relevance and applicability in various multimodal task applications, emphasizing the diverse nature of audio data within multimodal learning.

Similar to visual data, audio signals are a form of **"raw data"** that can be directly captured from the environment. However, unlike static images, audio signals possess **inherent continuity in the temporal dimension**.

### 2.2.1 Music

Music is a specific type of audio data playing a significant role in daily life. As a form of expressive art, music is considered the carrier and reflection of one's inner world.

**Genre taxonomy:** Traditional classical, symphony, modern pop, country music, etc.

**Narrative taxonomy:** Music is further classified into:
- **Diegetic music:** Integral to the narrative, existing within the story's universe and perceived by its characters (e.g., a character playing an instrument on screen).
- **Incidental music:** Intended solely for the audience's experience, underscoring emotions and scenes without being part of the story's world (e.g., a film score).

The music categorization can be subjective, with less strict and rigorous taxonomy for specific genres. One common feature: a music piece with high auditory quality has a **relatively large sampling rate**. For example, CD-quality music has a sampling rate of **44.1 kHz** [Dhariwal et al. 2020], producing over **2 million data points for a one-minute musical piece**. This high dimensionality imposes difficulties in data processing, driving the development of compact musical data representations.

#### Musical Data Representation Taxonomy

The survey classifies musical data representations into **"non-learning-based"** and **"learning-based"** categories:

**Non-Learning-Based Representations:**

*Continuous:*
- **Waveform:** The most general non-learning-based continuous data format. A 2D data structure depicting the sound pressure variation measured by air vibration in the time domain. Emphasizes temporal variations of audio signals.
- **Spectrogram (Mel-Spectrogram):** Compared to the waveform, the spectrogram also reflects the **frequency content of sounds over time**. In most cases, the waveform is referred to as the raw audio data.

*Discrete:*
- **1D Piano-Roll** [Dong et al. 2018]: A sparse data representation format where the horizontal axis represents timestamps and the vertical axis represents acoustic pitch.
- **2D MIDI (Musical Instrument Digital Interface):** Can be interpreted as a composed piano-roll format with instrument types, represented by different colors. Both 1D piano-roll and 2D MIDI can be decoded back to raw audio space by **pre-defined music synthesizers**.

**Learning-Based Representations:**

*Discrete:*
- **Vector Quantization (VQ):** Recent progress in deep learning introduced VQ [van den Oord et al. 2017; Razavi et al. 2019] to reduce high-dimensional data into a discrete token space. This is particularly leveraged in cross-modal music generation works (e.g., D2MGAN [Zhu et al. 2022 ECCV]; CDCD [Zhu et al. 2023 ICLR]).

*Continuous:*
- Neural networks such as CNNs encode raw audio signals to embedding features with desired dimensions, sharing similar attributes as visual data representations.

**Unique Musical Characteristics for Downstream Tasks:**
1. Musical data is a **sequence** where temporal coherence within a complete music piece must be emphasized.
2. In addition to the temporal dimension, audio data is characterized by **frequency features** in spectrogram form.
3. **Rhythm** is another important unique music characteristic for quality assessment.

### 2.2.2 Speech

Speech primarily refers to **audio signals of spoken languages**, closely related to natural languages with their intrinsic correspondence. Data representations for speech are similar to music (waveform and spectrogram in the non-learning-based category).

**Key uniqueness:** The natural **correlation to language**. Discrete representations of speech audio align with **language tokens**. This results in a more unified format compared to the learning-based VQ representation used in music audio — and influences architecture selection in methodology design.

**Classic Speech Tasks:**
- **Speech Separation:** Isolating individual speech tracks from a composite audio mix [Bahmaninezhad et al. 2019].
- **Automatic Speech Recognition (ASR)** [Yu & Deng 2016]: Converting spoken language into text. Essential for voice-activated interfaces and transcription services.
- **Multilingual Translation:** Due to the intrinsic correspondence between speech and language [Di Gangi et al. 2019].
- **Cross-Modal Translation (Audio↔Text):** [Chung et al. 2018].
- **Speech Generation from Visual Input:** Talking lip synthesis [Kim et al. 2021 NeurIPS].

**Sign Language — A Special Case:**
Sign language is a special type of speech-language for the hard-of-hearing and speech-impaired community. Unlike audible speech, it requires interpreting **visual signals from gestures**, exhibiting natural connections with visual and motion data. Research tasks include:
- **Sign Language Recognition:** Translating specific hand gestures into textual data [Bragg et al. 2019; Rastgoo et al. 2021].
- **Sign Language Generation:** The reverse process (text-to-gesture synthesis).
- Datasets [ASL Alphabet (Nagaraj 2018); How2sign (Duarte et al. 2021)] typically comprise visual data (images/videos) accompanied by language annotations.

### 2.2.3 Ambient Sound

Beyond speech and music, there are other audio signals accompanying certain events, referred to as **"ambient sound"** in this survey.

**Compared to music** (subjective) and **speech** (closely related to natural languages), ambient sound is more frequently encountered **in conjunction with videos** to characterize specific actions and events. Example: the sound of a crying baby naturally corresponds to a video showing the corresponding visual scenes. This unique correspondence enables ambient sound to provide **additional information to conventional video action recognition tasks**, leveraging the audio modality [Huh et al. 2023; Yang et al. 2017].

**Representation challenges:** In contrast to music (highly processed formats like MIDI) and speech (benefiting from natural text correspondence), the representation of ambient sound is **more ambiguous**. It lacks:
- Explicit formats like discrete token representations (unlike speech)
- Specific features like rhythm (unlike music)

These characteristics contribute to the **inherent ambiguity and challenges** in representing ambient audio, making it less structured than the other two audio subtypes.

---

## 2.3 Text

Text has been studied within the **Natural Language Processing (NLP)** community from the early years. The survey focuses on textual data types exhibiting close connections with other data modalities.

**Critical distinction:** Unlike visual and audio information (considered "raw data"), textual data **undergoes substantial processing**. It is a data type that has evolved through human civilizations, characterized by:
- A **highly unified format**
- **Precise semantics** despite linguistic differences
- **Highly informative and compact** representation (compared to the information redundancy in visual and audio signals)

A unique characteristic of text on the application side: the problem formulation of most NLP tasks can be **unified under the notion of "next word token prediction"**. This formulation represents a common underlying structure in various NLP tasks, contributing to the coherence and consistency within the field and its potential to solve multiple tasks via a **large foundation model** [Bommasani et al. 2021].

The NLP community witnessed significant attention with the remarkable success of **Large Language Models (LLMs)** such as **GPT-3** [Radford et al. 2019; Brown et al. 2020 NeurIPS].

### 2.3.1 Captions

Captions provide **sentence descriptions** that summarize either the entirety or a portion of the visual content in vision-and-text-related multimodal works. They may consist of single sentences or longer paragraphs of multiple sentences.

**Representation progression:**
- **Bag-of-Words (BOW)** [Mikolov et al. 2013]: Classic text representation format, representing a text corpus as a multiset (bag) of its words.
- **RNN-based (LSTM)** [Hochreiter & Schmidhuber 1997]: Captions processed by Recurrent Neural Networks with an internal memory state to obtain learning-based representations. The memory state design allows for recurrent connections among consecutive words to better interpret overall textual features.
- **BERT (Bidirectional Encoder Representations from Transformers)** [Devlin et al. 2018]: Large-scale pre-trained model for word embedding. A major breakthrough widely addressed in subsequent caption-related research studies.

### 2.3.2 Dialogue

Dialogue is a form of textual data in multimodal machine learning, **distinct from captions due to its inherently interactive nature**, which involves conversation among participants with logical coherence rather than a unilateral description of visual content.

**Special attention required:** Not only the words within a sentence (like plain captions), but also the **connections between different sentences** within a complete dialogue.

**Architectural implication:** In multimodal learning for vision and language, these unique dialogue characteristics are often addressed by constituting an additional data component — **history** — in the framework design. This component captures the flow of conversation (including prior exchanges) and is processed by dedicated mechanisms that operate alongside the processing module for individual sentences.

### 2.3.3 Question and Answer

A more specific categorization of textual data. While its representation is overall similar to other textual counterparts (tokens), Q&A pairs are often leveraged in **vision-language tasks** as an approach to:
- Study the **visual reasoning ability** of networks
- Evaluate specific task performance

**Visual Question Answering (VQA)** [Antol et al. 2015 ICCV] is the representative task that uses question and answer to reason about the visual context. Q&A is often closely related to dialogue, as interactions within dialogue can take the form of questions and answers.

### 2.3.4 ASR and OCR Text

While captions and dialogue are correlated with visual context in a high-level, semantic manner, **Automatic Speech Recognition (ASR)** and **Optical Character Recognition (OCR)** based texts present a **slightly different connection** to audio and visual information. Specifically:
- ASR and OCR are **foundational multimodal research topics** that have matured over decades.
- They exhibit **precise correspondence** between text and other data modalities [Gandhi et al. 2022; Javed et al. 2022].
- OCR additionally serves as a technique to obtain textual data from textual corpus.

---

## 2.4 Other Modalities

Multimodal learning encompasses a wide array of data modalities beyond vision, audio, and text. This survey focuses on modalities with **cognitive significance** that mirror the human perceptual system, grouping these under "others" and highlighting their relationships with primary modalities.

### 2.4.1 Graph

Graph data offers **structured representations of relational information** via nodes and edges, capturing connections and interactions between elements. While not a naturally occurring human-perception modality, it plays a significant role in machine learning when connected to other data modalities.

**Example:** Scene graphs establish graphical representations from images to interpret connections among objects. A typical multimodal application: **scene graph generation from visual context** [Li et al. 2017 ICCV; Lu et al. 2016 ECCV].

The non-Euclidean nature of graph data inspires **Graph Neural Networks (GNNs)** [Scarselli et al. 2008; Wu et al. 2020] — a powerful model architecture for processing graph data.

### 2.4.2 Optical Flow

Optical flow was first proposed in the last century as a measurement to characterize the **movement of objects in a visual scene** caused by the relative motion between an observer and a scene [Horn & Schunck 1981]. With deep learning, optical flow has been studied together with visual data [Teed & Deng 2020 (RAFT); Walker et al. 2015; Wang et al. 2019].

**Precise definition:** Optical flow is defined in a **pixel-level** manner via the change within consecutive image sequences.

**Challenge:** The calculation of optical flow is difficult because **environmental lighting** can impose large effects on pixel values of images.

Optical flow can be considered as a **specific motion presentation explicitly derived from visual information** — a processed, derivative representation rather than a raw data capture.

### 2.4.3 Point Clouds and Meshes

Both important forms of **3D data**, providing spatial and structural information that enriches understanding of physical environments.

- **Point Clouds:** Collections of vertices in a three-dimensional coordinate system.
- **Meshes:** Build on point clouds by connecting points with edges and faces, creating a comprehensive model that represents the shape and topology of 3D objects.

Like other "other" modalities, point clouds and meshes are not directly captured by sensory systems but are **constructed through processes that incorporate human insights** (e.g., LiDAR scanning, photogrammetry). Specialized architectures: **PointNet** [Qi et al. 2017 CVPR] for point sets.

### 2.4.4 Human-Centric Motions

Human motions are used to define various daily activities. They exist in multiple representation forms:

**2D Skeleton Data:**
- Captures keypoints of the human body, representing them as axis coordinates within an image.
- Extracted per-frame via pre-trained networks such as **OpenPose** [Cao et al. 2017 CVPR; Cao et al. 2019 IEEE TPAMI].
- Applicable in human-centered assistant systems (e.g., fall detection for elderly people).

**3D Human Motion Data:**
- Provides richer information with the extra depth dimension.
- Classic form: Incorporating depth information alongside conventional 2D keypoints.
- Alternative acquisition: Geometric reasoning from 3D data [Suwajanakorn et al. 2018 NeurIPS]; SLAM techniques applied to lidar sensor data [Streiff et al. 2021 ICRA].
- **SMPL (Skinned Multi-Person Linear Model)** [Loper et al. 2015 SIGGRAPH Asia]: A prevalent Computer Graphics (CG) representation that integrates skinning and blends shapes to represent human bodies.

**Comparison with Optical Flow:**
- Optical flow captures the motion of **all pixels** between two frames.
- Keypoint movement tracks **specific points of interest** across frames, allowing more focused analysis of object or feature dynamics.
- **3D video features** extend this analysis into the spatial domain, integrating depth information with motion for a richer, more detailed representation.

---

# 🔬 Core Technical Challenges in Multimodal Machine Learning

## Section 3: Multimodal Representation Learning

The multimodal representation learning field has undergone a **paradigm shift from conventional supervised representation to large-scale pretraining**. Classic supervised methods require fully annotated data, imposing limitations on training dataset size due to tedious human labeling. The trend has moved toward non-supervised settings using data that does not require human annotations — often collected from the Internet as naturally paired cross-modal data.

The primary research objective: **learning an effective and discriminative mapping between corresponding data representations from multiple modalities.**

---

## 3.1 Network Architectures

### 3.1.1 Convolutional Neural Networks (CNNs)

![CNN Architecture for Visual and Audio Representation](path/to/cnn_architecture.jpg)

CNNs [He et al. 2016 (ResNet); Krizhevsky et al. 2012 (AlexNet); Simonyan & Zisserman 2014 (VGG)] are widely adopted as backbone architectures in representation learning for **visual data**. The core idea: extract high-level data representations from raw data via complex functions composed of **convolutional layers and activation functions**.

The same idea has been adapted for learning **audio signal representations** [Hershey et al. 2017 ICASSP].

**Training objective for classification tasks:** Multi-class cross-entropy loss:

```
l = -∑(c=1 to N) y_c · log(p_c)
```

where `y_c` is the class label and `p_c` denotes the predicted probability. Features extracted from the last layer of CNNs are used as the actual data representation.

### 3.1.2 Recurrent Neural Networks (RNNs)

![RNN / LSTM Architecture for Sequential Text Processing](path/to/rnn_lstm_architecture.jpg)

A specific demand for learning data representations for natural languages is to consider **temporal correlations with sequential word order**. The NLP community follows a different vein of network architectures — **Recurrent Neural Networks (RNNs)** and **LSTM** [Graves 2013; Hochreiter & Schmidhuber 1997] — to address this challenge. Efforts have also been made to learn audio data representations via RNNs [Freitag et al. 2017].

### 3.1.3 Transformers

![Transformer Self-Attention Mechanism](path/to/transformer_attention.jpg)

Transformers [Vaswani et al. 2017 NeurIPS ("Attention is All You Need")] have gained great popularity in both computer vision and NLP. The core technical design: the **self-attention mechanism**, operating on sequential data to learn overall information.

**Advantages over CNNs and RNNs:**
1. **Flexibility** to deal with sequential data of variant lengths.
2. **Efficiency** through parallel computing (vs. RNNs' sequential processing).

Initially designed for NLP tasks [Brown et al. 2020 NeurIPS (GPT-3)], Transformers have been successfully applied in representation learning for:
- **Visual data** [He et al. 2022 CVPR (Masked Autoencoders)]
- **Audio data** [Truong et al. 2021 ICCV]

### 3.1.4 Mamba

**Mamba** [Gu & Dao 2023, arXiv:2312.00752] is a newer popular model showing promising downstream performance on **long language and audio sequence processing** compared to Transformers. Its key strength: addressing computational challenges by incorporating a **selective mechanism into state space models** — enabling linear-time sequence modeling.

### Language Model Pre-training Objective

For representation learning of textual data (Language Model Pre-training), the widely used problem formulation involves **"next token prediction"**, framing learning as a **joint conditional probability challenge**. The foundational optimization approach: maximizing the likelihood, typically achieved through cross-entropy loss.

---

## 3.2 Supervised Learning

Supervised setting requires **annotations from multimodal sources** to guide the learning process — the most classic representation learning setting [Desai & Johnson 2021 CVPR; Yuan et al. 2021 CVPR].

**Two widely adopted approaches:**

**Approach 1: Two-Stage Method**
"Representation learning in individual modality domain + mapping among modalities." Fixed backbone models perform first-stage feature extraction, followed by cross-modal mapping:
- **Collaborative Experts** [Liu et al. 2019]: Exploits existing pre-trained semantic embeddings from visual content; proposes collaborative experts model to aggregate multimodal information.
- **MEE (Mixture-of-Embedding-Experts)** [Miech et al. 2018]: Learns text-video embedding from heterogeneous data.
- **T2VLAD** [Wang et al. 2021 CVPR]: Focuses on global-local sequence alignment of video and textual representations.

**Approach 2: End-to-End Unified Representation**
Learns a unified representation for a given data pair in an end-to-end manner, freely optimizing feature extraction backbones:
- **Two-Branch Neural Networks** [Wang et al. 2018 IEEE TPAMI]: Proposes learning two-branch neural networks for image-text matching tasks.
- **Audio-Visual Correspondence** [Arandjelovic & Zisserman 2017 ICCV ("Look, Listen and Learn")]: Learns mutual representation via an "audio-visual correspondence" learning task.

---

## 3.3 Non-Supervised Learning

The terminology in non-supervised settings includes:
- **"Unsupervised":** Network training without human supervision.
- **"Weakly-supervised":** Supervisions may be noisy, limited, or imprecise.
- **"Self-supervised":** The model trains itself to learn one part of the input from another part of the input.

**Fundamental idea:** Relies on the premise of the **intrinsic synchronization nature among paired data from multiple modalities** [Radford et al. 2021 ICML; Yang et al. 2017 CVPR; Zhu et al. 2021 ICCASSP]. Examples: certain video actions naturally accompany characteristic sounds; images and captions are paired for training vision-language models.

**Two popular pre-training methods:**

### 3.3.1 Contrastive Learning Based

**CLIP (Contrastive Language-Image Pre-Training)** [Radford et al. 2021 ICML]:
- Trained on **400 million text-image pairs**.
- Considered one of the first large-scale pre-trained models for multimodal learning to bridge text and image data space.
- Follows the general idea of **aligning embedding space of paired images and corresponding textual descriptions**.
- Adopts the **batch construction technique** [Sohn 2016 NeurIPS] by encoding the entire sentence description as an entirety, instead of processing text word by word.
- Jointly trains a text encoder and an image encoder by **optimizing similarity scores** of given pairs.
- At inference: enables **zero-shot prediction** by embedding names or descriptions of target dataset classes in textual form via the learned text encoder.

**VATT (Video-Audio-Text Transformer)** [Akbari et al. 2021 NeurIPS]:
- A transformer-based self-supervised large-scale model for learning representations from raw video, audio, and text.
- Processes raw data from different modalities via **linear projection**.
- Trains via **Noise Contrastive Estimation (NCE)** to learn a semantically latent space.

### 3.3.2 Mask Reconstruction Based

Works include: [Chen et al. 2020 ECCV (UNITER); Lei et al. 2021 CVPR (ClipBERT); Lu et al. 2019 NeurIPS (ViLBERT)]. These follow the BERT [Devlin et al. 2018] paradigm of masked token prediction applied to multimodal inputs.

### 3.3.3 Critical Observations and Controversies

While large-scale models [CLIP; VATT; GPT-3] achieve impressive results, there are few radical novelties in model architecture and training techniques. Key controversies:
- Impressive results are **largely attributable to diverse and enormous carefully curated training data** and scale, rather than architectural innovation.
- **Privacy and ethical concerns** have been raised about training data composition.

Despite controversial voices, these models help build a more **unified toolkit connecting visual and textual spaces** in multimodal learning, promoting a large number of downstream works developed based on the aligned feature space.

---

## 3.4 Trend in Representation Learning

The machine learning and computer vision research community is rapidly advancing with a trend toward **scaling up data representation learning** using emerging foundation models, empowered by upgrades in dataset scale and computational resources.

Multimodal representations learned by large pre-trained models such as **CLIP** [Radford et al. 2021] have been successfully applied in various multimodal downstream tasks, boosting performance especially in terms of **generalization ability of models**.

**Critical caution:** Scaling up is **not a panacea**. Despite benefits, fundamental issues persist:
- **Out-of-distribution challenges**: Models may fail dramatically on data distributions not represented in training.
- **Amplified model bias** [Wang & Russakovsky 2023 ICCV]: Large-scale training data amplifies demographic, cultural, and representational biases embedded in internet-scraped corpora.

Future research needs to focus on **real-world scenarios with more edge cases and complex data formats** for safe and responsible deployment.

---

# Section 4: Discriminative Applications

For discriminative applications, popular approaches inherit neural networks from general representation learning (Section 3.1), with additional modules to adapt to task-specific objectives.

**General methodological design in multimodal learning:**
1. **Separate Processing:** Data from different modalities are first processed with respective network branches.
2. **Unified Fusion:** Inter-modality learning is performed by extra mutual modules before outputting final results.

---

## 4.1 Vision + Audio (Discriminative)

### 4.1.1 Audio-Visual Event Localization (AVEL)

![AVEL Pipeline: Cross-Modal Attention for Temporal Localization](path/to/avel_pipeline.jpg)

An **Audio-Visual Event (AVE)** is defined as an event that is **both audible and visible** in a video segment [Tian et al. 2018 ECCV]. The **AVEL task** aims to localize the AVE within an unconstrained video [Duan et al. 2021 WACV; Lin & Wang 2020 ACCV; Tian et al. 2018 ECCV; Xuan et al. 2020 AAAI; Yu et al. 2021].

**First proposed in:** [Tian et al. 2018 ECCV], together with the AVE video dataset.

**Task objective:** Resembles action recognition with ambient audio data, with additional requirement for **temporal localization** under supervised or weakly-supervised settings.

**Common method:** Achieving cross-modal interactions via different **attention modules** [Duan et al. 2021; Tian et al. 2018; Xuan et al. 2020; Yu et al. 2021].

**Framework:** Processes audio and visual data with **separate encoders**, fuses the processed information for temporal localization and activity classification. The temporal connection from video streams is often addressed via **LSTM** [Hochreiter & Schmidhuber 1997].

**Evaluation:** Prediction accuracy metric.

### 4.1.2 Audio-Visual Video Parsing (AVVP)

The **AVVP problem** aims to **parse a video into temporal segments** and label them as either audible, visible, or both [Lin et al. 2021 NeurIPS; Tian et al. 2020 ECCV; Wu & Yang 2021 CVPR].

**Distinction from AVEL:** AVVP emphasis is on **recognition**, while AVEL focuses more on **temporal localization**.

**Representative works:**
- **Lin et al. [2021 NeurIPS]:** Introduced sequence-to-sequence integration of audio and visual features.
- **Yu et al. [2021]:** Explored AVVP by taking potential **audio-visual asynchrony** into account.

### 4.1.3 Visual Sound Source Localization (VSSL)

![VSSL: Localizing Sound Sources in Visual Scenes](path/to/vssl_pipeline.jpg)

**VSSL task** aims to locate the **corresponding visual locations within an image given the sound** [Oya et al. 2020 ACCV; Qian et al. 2020 ECCV; Rachavarapu et al. 2021 ICCV; Senocak et al. 2018 CVPR; Senocak et al. 2022 WACV; Song et al. 2022 CVPR].

**Historical origin:** Deep learning based visual localization was first proposed in [Senocak et al. 2018 CVPR]. The original sound source localization (SSL) task was widely studied in signal processing [Grumiaux et al. 2022].

**High-level idea:** Learning correlations between paired audio and visual data; the VSSL task switches the **regions of interest within visual data** given different ambient audio signals.

**Overall pipeline:** Separate encoders for visual and audio input, fusing audio-visual information for learning a localization module during training. Technical differences lie in:
- Fusion via **attention mechanisms** [Qian et al. 2020; Senocak et al. 2018]
- Training technique using **localization or contrastive loss** [Qian et al. 2020; Rachavarapu et al. 2021; Senocak et al. 2018]

**Evaluation metrics:** **cIoU (Complete IOU)** and **AUC (Area under the ROC Curve)** scores to quantify precision of predicted areas for sound sources.

---

## 4.2 Vision + Text (Discriminative)

### 4.2.1 Visual Grounding

![Visual Grounding: Two-Stage vs. One-Stage Frameworks](path/to/visual_grounding_pipeline.jpg)

**Visual grounding** aims to **localize the object within an image given a text description as input** [Deng et al. 2018 CVPR; Deng et al. 2021 ICCV; Fukui et al. 2016; Hong et al. 2019; Huang et al. 2022 AAAI; Li et al. 2020 ICML; Liu et al. 2018 ICCV; Liu et al. 2020 AAAI; Liu et al. 2021 CVPR; Sun et al. 2021 IEEE TPAMI; Yang et al. 2019 CVPR; Yang et al. 2020 ECCV; Yu et al. 2016 ECCV; Yu et al. 2018].

**Historical origin:** Cross-reference between sentences and visual context first proposed and studied in [Karpathy et al. 2014 NeurIPS], where the task was also known as **"referring expression comprehension"**.

**Task evolution:**
- **Pioneering works** [Hu et al. 2016 CVPR; Mao et al. 2016 CVPR; Yu et al. 2016 ECCV]: Grounding of a single object from description sentence input; region of interest expected to achieve maximum posterior probability for given textual description.
- **More recent works:** Tackle more challenging setting refined into two sub-objectives: (1) **phrasing** — localizing all objects mentioned in textual description; (2) **grounding** — individually detecting corresponding boxes in the image [Liu et al. 2021 CVPR; Plummer et al. 2017 ICCV].

**Methodology: Learning Settings:**
- **Supervised:** Phrase-object pair annotation provided.
- **Weakly Supervised:** Phrase annotations for textual description input removed.
- **Unsupervised:** Annotations for both data modalities completely removed.

**Methodology: Framework Types:**
- **Two-Stage:** Extract region proposals for potential objects → rank and match proposals with language phrases.
- **One-Stage:** Visual objects and textual phrases aligned and connected during learning to avoid redundant region proposals.

**For weakly/unsupervised settings:** Extra regularization losses (structural loss, discriminative loss [Sun et al. 2021; Xiao et al. 2017 CVPR]) needed to learn correlations between object regions and textual phrases.

**Evaluation metrics:**
- **IoU (Intersection over Union)** between predicted and ground truth boxes; threshold value of 0.5.
- **PointIt (pointing game metric)** [Xiao et al. 2017 CVPR]: Computes pixel location with maximum predicted attention weight; if the selected hit point lies within the ground truth box area, the prediction is counted as valid.

### 4.2.2 Temporal Activity Localization (TAL)

Also known as **video grounding**. **TAL** seeks to locate the **temporal segment of a video clip given the language description of a certain activity as query** [Anne Hendricks et al. 2017 ICCV; Chen & Jiang 2019 AAAI; Chen & Jiang 2020 ECCV; Gao et al. 2017 ICCV; Wang et al. 2019 CVPR; Zhang et al. 2021 CVPR; Zhang et al. 2021 CVPR].

**Distinction from visual grounding:** TAL requires additional **reasoning and matching along the temporal direction**, requiring models to not only capture correlations between visual activity and language, but also **temporally localize the segment** among consecutive video frames.

**High-level framework:** Similar structure to previous multimodal discriminative tasks — separate encoders, multimodal processing module for fusing features, decoder module adapted for specific task objectives.

**Representative early work: CTRL (Cross-modal Temporal Regression Localizer)** [Gao et al. 2017 ICCV]: Proposes a temporal localization regression network to align the fused visual-textual information with video temporal locations.

**Evaluation metrics:** **Mean IoU** and **IoU@a** (where `a` stands for the percentage of overlap between predicted segment and ground truth annotations).

### 4.2.3 Visual Entailment (VE)

**VE** seeks to **predict the logical relationship of a piece of text to an image** [Thomas et al. 2022 ECCV; Xie et al. 2018 arXiv; Xie et al. 2019 arXiv].

**Historical origin:** Developed from the **textual entailment task** [Dagan et al. 2005], whose initial objective was to decide if a hypothesis can be logically deduced from a premise. [Xie et al. 2018] extends textual entailment to the multimodal context, replacing the textual premise with an image.

**Task refinement:** [Thomas et al. 2022 ECCV] further refines the task by introducing different levels of granularity.

**Task emphasis:** **Multimodal reasoning ability** of networks.

**Methodology:**
- Early methods [Xie et al. 2018; Xie et al. 2019]: Separate network branches to process visual and textual data, leveraging attention interaction.
- Refined framework [Thomas et al. 2022]: Disentangles textual hypothesis into constituent parts; proposes enhancing reasoning by introducing an **Abstract Meaning Representation (AMR) graph** for decomposed textual components.

**Evaluation:** Prediction accuracy given the premise as input.

### 4.2.4 Spatio-Temporal Video Grounding (STVG)

![STVG: Spatio-Temporal Tube Prediction](path/to/stvg_pipeline.jpg)

**STVG** is a recent multimodal task at the intersection of visual grounding and temporal localization, integrating reasoning among **space, time, and language** within video context [Jin et al. 2022 NeurIPS; Su et al. 2021 ICCV; Yang et al. 2022 CVPR; Zhang et al. 2020 CVPR].

**Precise task definition:** Given an untrimmed video and a textual description of an object, the task seeks to **localize a spatio-temporal tube** (i.e., a sequence of bounding boxes) for the target object described.

**Methodology paradigms:**
- **Two-Stage:** Leverages pre-extracted object proposals, integrates temporal localization via attention mechanisms [Wang et al. 2019 CVPR; Yang et al. 2019 ICCV].
- **One-Stage:** Does not rely on prior object proposals [Kamath et al. 2021 ICCV (MDETR); Yang et al. 2022 CVPR (TubeDetr)].

**Network architecture:** Transformers are widely adopted as the backbone [Su et al. 2021; Yang et al. 2022; Zhang et al. 2020].

**Evaluation:** IoU metrics comparing frame overlap between ground truth and predicted timestamps.

---

## 4.3 Vision + Audio + Text (Discriminative)

### 4.3.1 Multimodal Retrieval

**Multimodal retrieval** operates on the representation space by **measuring similarities among learned representations from different modalities** [Chun et al. 2021 CVPR; Gu et al. 2018 CVPR; Wang et al. 2017 ACMM; Wang et al. 2016 arXiv; Wang et al. 2021 CVPR; Wang et al. 2022 arXiv; Zhen et al. 2019 CVPR; Zhu et al. 2021 ICCASSP].

Retrieval is one of the most frequently used **downstream tasks in representation learning works**. Multimodal retrieval extends to **cross-modality scenarios**: retrieving items that match the input from a different data modality (e.g., text-based vision retrieval, audio-based vision retrieval).

**Representative works:**
- **CAMP** [Wang et al. 2019 ICCV]: Learns text and image embedding via **message passing across modalities**.
- **Gu et al. [2018 CVPR]:** Improves text-visual retrieval with auxiliary generative models.
- **T2VLAD** [Wang et al. 2021 CVPR]: Text-based video retrieval via global-local alignment method.
- **Zhu et al. [2021 ICCASSP]:** Learns mutual audio-visual latent space via VAE-based framework for audio-visual cross-modality retrieval.
- **Oncescu et al. [2021 INTERSPEECH]:** Proposes to retrieve audio signals given natural language queries.

### 4.3.2 Audio-Visual Question Answering (AVQA)

**AVQA** is an extension of visual question answering with integrated audio modality [Li et al. 2023 ACMM; Li et al. 2022 CVPR; Yang et al. 2022 ACMM; Yun et al. 2021 ICCV]. AVQA often involves questions regarding different visual objects, sounds, and their associations in videos.

**Framework:** Extended from VQA frameworks with extra interactions with audio data. An intuitive framework [Li et al. 2022 CVPR] expands the two-branch encoder design into **three-branch**, separately processing video, audio, and textual data before introducing interactions via the attention mechanism.

**Evaluation:** Answer prediction accuracy.

---

# Section 5: Generative Applications

Focus: **Cross-modality synthesis tasks**. These involve generating data from a specific modality or multiple modalities as input.

**Two high-level generation approaches:**
1. **Retrieval-based generation:** Searching for an item or several items that most resemble the "generated" data. Core idea follows similarity measurement on data representation level. The survey categorizes purely retrieval-based works under Representation Learning.
2. **Direct synthesis:** Truly generating data via neural networks — the primary focus of this section.

---

## 5.1 Generative Networks

### 5.1.1 VAE-Based Models

![Variational Autoencoder Architecture](path/to/vae_architecture.jpg)

**Variational Autoencoders (VAEs)** [Kingma & Welling 2014 ICLR] are classic generative models based on deep neural autoencoders [Hinton & Salakhutdinov 2006] under the unsupervised learning setting. The core premise: an effectively trained encoder should learn data representation such that encoded representations can be **decoded to reconstruct the original data input**.

**Difference from conventional autoencoders:** VAEs introduce **regularizations on the bottleneck level** by re-parameterizing the latent space using **Gaussian priors**. The learned Gaussian parameters allow sampling new data.

**Training objective — two types of losses:**

The classic variational objective is derived from:

```
log p(x) = E_{q(z|x)}[log p(x|z)] - D_KL[q(z|x) || q(z)]
```

where `p` represents the decoder, `q` is the encoder, `x` denotes the original raw data, and `z` denotes the learned latent embedding.

The re-parameterization technique assumes `z ~ N(μ, σ)`, enabling minimization of KL divergence via sampling from N samples:

```
∑(i=1 to n) [σ²_i + μ²_i - log(σ_i) - 1]
```

**Typical losses:**
- **Variational loss (ELBO):** Regularization loss on latent representation space (Kullback–Leibler divergence).
- **Reconstruction losses:** On output data (e.g., Mean Square Errors — MSE).

**Applications:** Widely exploited in generative tasks in audio and images [Higgins et al. 2017 ICLR (β-VAE); Kingma & Welling 2014]; in multimodal context for cross-modality generations [Spurr et al. 2018 CVPR; Zhu et al. 2021 ICCASSP].

### 5.1.2 GAN-Based Models

![Generative Adversarial Network: Generator vs. Discriminator](path/to/gan_architecture.jpg)

**Generative Adversarial Networks (GANs)** [Goodfellow et al. 2014 NeurIPS] are a mainstream backbone type for generative models. **Two agents** play an adversarial game:
- **Generator G:** Aims to synthesize realistic data resembling real data to fool the discriminator.
- **Discriminator D:** Aims to distinguish synthesized data by G from real data.

Similar to VAEs, GAN training does **not require external annotations** — only real raw data — making it frequently used in unsupervised or weakly-supervised settings.

**Standard training minimizes losses from two aspects:**
1. Latent space regularization (adversarial losses)
2. Reconstruction optimizations

**Classic GAN loss:**

```
min_G max_D V(D, G) = E_{x~p_data(x)}[log D(x)] + E_{z~p_Z(z)}[log(1 - D(G(z)))]
```

where G and D are the generator and discriminator, x denotes original raw data, and z denotes the learned latent embedding.

**GAN variants:**
- **Wasserstein GAN** [Arjovsky et al. 2017 ICML; Gulrajani et al. 2017 NeurIPS]: Uses Wasserstein loss for more stable training.
- **Conditional GAN** [Mirza & Osindero 2014 arXiv]: Conditions generation on additional input information.

**Application progression:**
- First widely applied in **image generation** [Brock et al. 2018 arXiv (BigGAN); Isola et al. 2017 CVPR (Pix2Pix)]
- Later explored for **audio synthesis** [Kong et al. 2020 NeurIPS (HiFi-GAN); Kumar et al. 2019 NeurIPS (MelGAN)]
- Cross-modality areas [Zhan et al. 2019 CVPR; Zhu et al. 2022 ECCV (D2MGAN); Zhu et al. 2019 CVPR (DM-GAN)]

### 5.1.3 DPM-Based Models

![Diffusion Probabilistic Model: Forward and Reverse Markov Chain](path/to/dpm_architecture.jpg)

**Diffusion Probabilistic Models (DPMs)** [Sohl-Dickstein et al. 2015 ICML] are another type of generative backbone that have become very popular in recent years.

**Principal mechanism:** A Markov chain of finite steps in two opposite directions:
- **Forward direction (Diffusion process):** Gradually adds noises to given data at each diffusion step.
- **Inverse direction (Denoising process):** Removes the noises added in forward steps and recovers the actual data from a non-informative noisy distribution.

**Two variants of conventional DPMs:**
1. **Continuous state space:** Parameterizes the diffusion process with Gaussian noises [Dhariwal & Nichol 2021 NeurIPS; Ho et al. 2020 NeurIPS (DDPM); Ho et al. 2022 NeurIPS (Cascaded); Kingma et al. 2021 NeurIPS; Nichol & Dhariwal 2021 ICML; Song & Ermon 2020 NeurIPS; Song et al. 2020 ICLR; Song et al. 2021 NeurIPS].
2. **Discrete state space:** Formulates diffusion process with state transition matrix [Austin et al. 2021 NeurIPS; Gu et al. 2022 CVPR; Zhu et al. 2023 ICLR (CDCD)].

**Training loss:** Variational lower bound (Lvb):

```
L_vb = E_q [ D_KL(q(x_T|x_0) || p(x_T))   [L_T]
           + ∑_{t>1} D_KL(q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t))   [L_{t-1}]
           - log p_θ(x_0|x_1)   [L_0]
           ]
```

where `q` and `p` represent the diffusion and denoising processes; `x_t` denotes the data at diffusion step `t`.

**Additional practical losses:**
- Auxiliary loss [Austin et al. 2021; Gu et al. 2022]
- Classifier-free guidance [Ho & Salimans 2022 arXiv]
- Contrastive diffusion losses [Zhu et al. 2023 ICLR (CDCD)]

**Applications:** Images [Dhariwal & Nichol 2021; Ho et al. 2020; Ho et al. 2022; Hu et al. 2021; Nichol & Dhariwal 2021; Song & Ermon 2020; Song et al. 2020]; audio [Kong et al. 2020 ICLR (DiffWave); Lee & Han 2021; Mittal et al. 2021]; cross-modality: text-to-image [Gu et al. 2022; Nichol et al. 2021 (GLIDE); Zhu et al. 2023 (CDCD)]; dance-to-music [Zhu et al. 2023 (CDCD)].

---

## 5.2 Vision + Audio (Generative)

### 5.2.1 Music Generation from Vision

Recent studies on generating music from visual data (usually videos) are categorized by adopted music representations:

**Branch 1: Symbolic Audio Representations (1D Piano-Roll, 2D MIDI)**

Works: [Aggarwal & Parikh 2021 (Dance2Music); Di et al. 2021 ACMM; Gan et al. 2020 ECCV (Foley Music); Su et al. 2020 NeurIPS (AUDEO)].

**Advantages:**
- Symbolic representations decoded to raw audio by **pre-defined synthesizers** with no additional noise, ensuring high-quality generated music.
- **Lower computational cost** — symbolic representations are very sparse and low-dimensional, facilitating learning and inference.

**Disadvantages:**
- Restricted in terms of **music diversity and flexibility**.
- Generated music typically limited to specific pre-defined instrumental sounds.
- Most symbolic-based music generation works trained based on ground truth MIDI annotations via **cross-entropy loss** (not directly using generative backbones from Section 5.1).

**Branch 2: Learning-Based Musical Representations (VQ)**

Works: [D2MGAN (Zhu et al. 2022 ECCV); CDCD (Zhu et al. 2023 ICLR)].

Most recent cross-modal music generations adopt the **discrete form of learned musical feature — Vector Quantization (VQ)** — as intermediate representations, leveraging the large-scale pretrained music synthesis model **JukeBox** [Dhariwal et al. 2020 arXiv].

**D2MGAN** [Zhu et al. 2022 ECCV]: GAN-based framework taking human body motion data and dance video frames as input; generates musical VQ representation.

**CDCD** [Zhu et al. 2023 ICLR]: Builds upon diffusion probabilistic model with a **discrete state space represented by VQ**; incorporates a **contrastive diffusion loss** to train the network to improve input-output correspondence for cross-modality applications.

### 5.2.2 Speech Generation from Videos

![Lip-to-Speech Synthesis Pipeline](path/to/lip_to_speech.jpg)

Specific generation task: **synthesizing speech audio from videos of human speaking** [Ephrat et al. 2017 ICCV Workshops; Ephrat & Peleg 2017 ICASSP; Kim et al. 2021 NeurIPS; Michelsanti et al. 2020; Mira et al. 2022; Prajwal et al. 2020 CVPR; Salik et al. 2019 AAAI; Vougioukas et al. 2019 arXiv; Yadav et al. 2021 ICASSP].

**Unique aspect:** Speech largely relies on the **movement of lips** while speaking. Based on this, many works focus on reading and interpreting **visual lip movement from video input** and converting it to audio waveform — explaining why this "video-to-speech" synthesis task is also known as **"lip-to-speech" generation**.

**Key insight:** Despite the topic of audio generation from videos, a large percentage of works focus on **"motions in the videos"** rather than raw videos.

**Technical approaches:**
- **Cross-modal audio-visual attention mechanisms** to enhance correlations between lip movement and speech audio.
- **Kim et al. [2021 NeurIPS]:** Attentional GAN with visual context to read lips for speech synthesis.
- **Yadav et al. [2021 ICASSP]:** VAE generative backbone with stochastic modelling approach.
- **Prajwal et al. [2020 CVPR]:** Disentangled speech features including individual speaking styles.

### 5.2.3 Ambient Sound Generation from Videos

Research works: [Chen et al. 2018 ECCV Workshops; Chen et al. 2020 IEEE TIP; Zhou et al. 2018 CVPR].

Special emphasis: **Alignment between generated sound and visual context** — both **semantic** and **temporal** alignments.

**Representative works:**
- **Chen et al. [2018 ECCV Workshops]:** Addresses semantic alignment problem using **perceptual loss** and considering sound categories in the optimization process.
- **Zhou et al. [2018 CVPR]:** Classic encoder-decoder framework for video input and audio decoder; proposes three method variants: frame-to-frame, sequence-to-sequence, and flow-based.
- **REGNET** [Chen et al. 2020 IEEE TIP]: Tackles both semantic and temporal alignment; core technical design includes a visual encoder and an **audio forwarding regularizer**.

**Observation:** Compared to speech and music, ambient sound has relatively fewer unique attributes beyond its correspondence with certain activities. High-level technical ideas are rather general and resemble standard pipeline designs.

### 5.2.4 Visual Generation from Sound

**Specific sub-field:** Synthesizing **talking faces from speech audio** [Chen et al. 2018 ECCV (Lip Movements); Chen et al. 2019 CVPR (Hierarchical); Das et al. 2020 ECCV; Song et al. 2018 arXiv; Vougioukas et al. 2020 IJCV; Zhang et al. 2021 CVPR; Zhou et al. 2021 CVPR].

**Input for talking face generation:** Reference image + driving audio track.

**Framework evolution:**
- **Early works** [Chen et al. 2018 ECCV; Song et al. 2018; Vougioukas et al. 2020]: General pipeline with two separate encoders for input and decoder for synthesizing talking videos, mostly via **GAN-based generative backbones**.
- **More recent works:** Refine and improve synthesis results by splitting previous architecture into **hierarchical structures** [Chen et al. 2019 CVPR; Das et al. 2020 ECCV].
- **Flow-based approaches:** Specific motion data (optical flow) used to enable high-resolution generations [Zhang et al. 2021 CVPR].
- Some works reformulate video generation as a **motion generation task in the form of optical flow** [Chatterjee & Cherian 2020 ECCV; Denton & Fergus 2018 ICML].

---

## 5.3 Vision + Text (Generative)

### 5.3.1 Caption Generation from Vision

**Image and video captioning** [Chen et al. 2019 NeurIPS; Chen et al. 2017 CVPR; Lu et al. 2017 CVPR; Vinyals et al. 2015 CVPR; Wang et al. 2018 CVPR; Wang et al. 2017 NeurIPS; Wu et al. 2022 IEEE TPAMI; You et al. 2016 CVPR] aims to **generate a language textual description of given visual data**.

**Framework evolution:**
- **Early deep learning era** [Mao et al. 2014 arXiv; Vinyals et al. 2015 CVPR]: General encoder-decoder pipeline with **CNNs** (image encoder) and **RNNs** (text decoder).
- **Attention era:** Attention mechanism [Lu et al. 2017 CVPR; Xu et al. 2015 ICML; You et al. 2016 CVPR] to enhance correlations between sentence descriptions and corresponding visual concepts.
- **Other techniques:** Adversarial learning [Dai et al. 2017 ICCV]; reinforcement learning [Liu et al. 2017 ICCV].

**Novel task settings:**
- **Audio-Augmented Captioning** [Alamri et al. 2019 CVPR; Hori et al. 2019 ICASSP; Schwartz et al. 2019 CVPR]: Incorporates ambient audio from video as additional input modality.
- **Multi-Agent Dialog Captioning** [Zhu et al. 2020 ECCV; Zhu et al. 2021 IEEE TPAMI]: Proposes new setting where part of visual data becomes inaccessible as input; proposes a dialog process between two agents as supplement to missing visual input.

### 5.3.2 Dialog Generation from Vision

Two sub-categories:

**Visual Question Answering (VQA)** [Agarwal et al. 2020 CVPR; Antol et al. 2015 ICCV; Chen et al. 2020 CVPR; Das et al. 2017 CVPR; Goyal et al. 2017 CVPR; Shih et al. 2016 CVPR; Xiao et al. 2021 CVPR]:
- Answers a **single question** related to visual input.
- First introduced in [Antol et al. 2015 ICCV].
- Mainstream frameworks: encoder-decoder pipeline with separate encoders for visual and textual data, decoder for generating answer words.
- **Critical bias problem:** Models may not truly rely on visual context to answer questions. Example: "What is the color of the sky?" likely answered "blue" regardless of image content. Recent works address spurious patterns and bias by analyzing **causal relations** [Chen et al. 2020 CVPR; Niu et al. 2021 CVPR].

**Visual Dialog** [Das et al. 2017 CVPR; De Vries et al. 2017 CVPR; Jain et al. 2018 CVPR; Qi et al. 2020 CVPR; Seo et al. 2017 NeurIPS; Wu et al. 2018 CVPR; Zhu et al. 2020 ECCV; Zhu et al. 2021 IEEE TPAMI]:
- Maintains **multiple rounds of question-answer interactions** with internal logic.
- First proposed in [Das et al. 2017 CVPR]; general encoder-decoder framework extended with an additional encoder for **dialog history data**.
- Challenge: Multiple rounds of Q&A may refer to different parts of visual context, requiring precise referencing of key information.
- Bias issue from VQA remains.

### 5.3.3 Image Synthesis from Text

![Text-to-Image Generation Pipeline: GAN → Auto-Regressive → Diffusion](path/to/text_to_image_pipeline.jpg)

A very active research topic since 2016 [Avrahami et al. 2022 CVPR; Gu et al. 2022 CVPR; Li et al. 2019 CVPR; Liu et al. 2022 ECCV; Nichol et al. 2021 (GLIDE); Ramesh et al. 2021 ICML (DALL·E); Ramesh et al. 2022 (DALL·E 2); Rombach et al. 2022 CVPR (Stable Diffusion); Xu et al. 2018 CVPR (AttnGAN); Zhang et al. 2017 CVPR (StackGAN); Zhang et al. 2018 IEEE TPAMI (StackGAN++); Zhu et al. 2023 ICLR (CDCD)].

**Chronological stage progression:**

**Pre-2020: GAN-Based Approaches** [Reed et al. 2016 ICML; Zhang et al. 2017 CVPR (StackGAN); Zhang et al. 2018 IEEE TPAMI (StackGAN++); Xu et al. 2018 CVPR (AttnGAN)]:
Mainstream GAN architectures for text-to-image generation.

**2020+: Auto-Regressive Models** [Ding et al. 2021 NeurIPS (CogView); Ramesh et al. 2021 ICML (DALL·E)]:
Adaptation of language processing techniques (Auto-Regressive models) for image generation.

**2020+: DPM-Based Methods** [Dhariwal & Nichol 2021 NeurIPS; Ho et al. 2020 NeurIPS (DDPM); Ho et al. 2022 (Cascaded); Gu et al. 2022 CVPR; Nichol et al. 2021 (GLIDE); Nichol & Dhariwal 2021 ICML; Rombach et al. 2022 CVPR; Zhu et al. 2023 ICLR (CDCD)]:
Gradually became most active methods for vision generation tasks due to impressive performance and better tractability compared to GANs.

**Large-Scale Models:**

**DALL·E** [Ramesh et al. 2021 ICML]: Two separate stages:
1. Stage 1: Learns visual concepts from images via **discrete VAE**.
2. Stage 2: Fuses learned discrete image embedding with textual tokens to train a **transformer** [Vaswani et al. 2017].
At inference: Obtains fused text-image embedding for given text description and image candidates; uses pre-trained **CLIP for image re-ranking** to get generated images with higher similarities.

**DALL·E 2** [Ramesh et al. 2022]: Improved text-to-image generator; uses **Diffusion Probabilistic Models (DPMs)** on the image embedding space from CLIP for image synthesis.

### 5.3.4 Text-Guided Image Editing

A step further from image synthesis: **performing editing based on text prompt for given raw images** [Aghajanyan et al. 2022; Avrahami et al. 2022 CVPR (Blended Diffusion); Kim et al. 2022 CVPR (DiffusionCLIP); Kwon et al. 2023 ICLR; Nichol et al. 2021 (GLIDE); Pan et al. 2023 (DragGAN); Tang et al. 2022; Zhu et al. 2023 NeurIPS].

**Task objectives (two-fold):**
1. Achieve the **target editing effect**.
2. **Preserve remaining features** of the given images.

**Methodology categories (similar to T2I, but with raw image as additional input):**

**GAN-based:** [Pan et al. 2023 (DragGAN); Zhu et al. 2020 ECCV].

**DPM-based:** [Avrahami et al. 2022 CVPR; Kim et al. 2022 CVPR; Nichol et al. 2021; Zhu et al. 2023 NeurIPS].

DPM-based image editing methods categorized by **whether additional learning is required given a pre-trained generative model:**
- **Fine-tuning required:** Fine-tuning parameters of given pre-trained models [Kim et al. 2022 CVPR (DiffusionCLIP)]; learning extra neural network modules [Kwon et al. 2023 ICLR].
- **Learning-free:** Explicitly leveraging intrinsic abilities of DPMs in exhibiting semantics along generation trajectories [Zhu et al. 2023 NeurIPS (Boundary Guided)].

### 5.3.5 Video Synthesis from Text

Works: [Ho et al. 2022 arXiv (Video Diffusion); Hu et al. 2022 CVPR; Li et al. 2018 AAAI].

**Compared to image synthesis:** Videos consist of multiple consecutive frames that are **temporally and spatially correlated**, in addition to pixel-level computationally extensive synthesis for a single visual frame.

**Representative approaches:**
- **Li et al. [2018 AAAI]:** VAE or GAN-based generative backbones with language priors.
- **Hu et al. [2022 CVPR]:** Modifies task formulation by providing a **reference image** and a textual description; synthesizes videos based on given input.
- **Video Diffusion Model** [Ho et al. 2022 arXiv]: Modified **3D U-Net architecture** equipped with an extra temporal dimension.
- **Make-A-Video:** One of the first large-scale video generation models. Leverages recent advances in text-to-image field to learn visual information; proposes learning temporal motions from **unlabeled large-scale video data**. At inference, composes visual content with learned motions to synthesize a realistic video.

---

# Section 6: Further Discussion

## 6.1 Insights from Data and Methodology Design

The survey revisits and discusses the correlation between data nature and methodology design from two aspects: **semantics of data modalities** and **specific formats**.

### 6.1.1 Data Semantics Axis

**Raw information sources** (visual data, ambient sound): Sensory information directly captured from the environment — usually high-dimensional, can be further processed with information redundancy.

**Processed information** (text data, speech): These modalities have undergone extensive processing throughout the evolution of human civilization — they already possess **meaningful semantics** and are **more information compact** with a unified representation in the form of tokens.

**Implication for NLP:** The highly processed nature of text data and its consistent problem formulation ("next word token prediction") paved the way for large-scale foundational models (GPT-3 family). The unified nature of text allows application to various tasks without significant modifications.

**Implication for Computer Vision:** Visual data (raw information sources) requires **extensive representation learning** and specific downstream application stages. The complexity and diversity of visual tasks make it more challenging to develop a unified foundational model. Researchers constantly explore novel representation learning techniques and task-specific approaches.

**Implication for Audio:** Researchers follow either community (NLP or CV) based on the **specific audio type** and task requirements.

### 6.1.2 Data Format Axis

**Continuous data** (images, ambient sound): Continuity in spatial or temporal dimensions benefits from model architectures like **CNNs** specifically designed to handle spatial and temporal dependencies, capturing local and global correlations.

**Discrete format data** (MIDI musical representations, textual word tokens): Models like **Transformers** are more appropriate for modeling dependencies among discrete elements.

---

## 6.2 Future Directions and Challenges

### 6.2.1 Observations on Discriminative Tasks (Vision + Audio)

Existing works largely follow the general pipeline: **separate data encoders → cross-modality attention feature fusions → decoder module** for various task objectives.

**Key gap:** All existing works process ambient audio data **as an entirety** without specifically looking into acoustic features of audio signals. Example: certain ambient audio signals may include higher pitch and frequency than others — a strong supplementary indicator for purely vision-based recognition that current discriminative works **do not exploit**.

In contrast, existing audio-involved **generative works** have explored more disentangled features such as **rhythms, pitches, and genres** for both synthesis and editing purposes.

### 6.2.2 Observations on Discriminative Tasks (Vision + Text)

Representative early classic methods use the **LSTM model** to deal with textual language data with word orderings. Later, the success of the Transformer model encouraged a **fast technical transition from LSTM to Transformers** for the text processing branch in multimodal learning.

### 6.2.3 Two Primary Future Research Directions

**Direction 1 — Unified General Model:**
Seeking to establish a **unified and general model** that efficiently learns representations of all modalities of interest. Similar to large-scale pre-training models (Section 3.3), such a model should greatly help with:
- Various downstream applications
- Specific cross-modality generations
- Interactive editing
- Evaluations

**Direction 2 — Fine-Grained Specific Applications:**
With increasing demand for more fine-grained and detailed applications in daily life, developing and achieving better performance for more **specific and crafted tasks** remains a crucial direction.

### 6.2.4 Human Intervention

A possible future direction: **human intervention for ultimate multimodal perception AI systems**. As the ultimate objective is to bring intelligence to machines as real humans, human intervention could be a critical part to guide the general research direction. Concrete example: involving humans to provide more controls on cross-modality generations and downstream tasks such as editing [Menapace et al. 2022 CVPR; Menapace et al. 2021 CVPR].

---

# 🏗️ System Architecture & Visual Pipeline

## Primary Model Architecture

![Model Architecture](IMG_20260612_151314.jpg)

The figure above illustrates the primary multimodal learning architecture — a multi-branch encoder pipeline with cross-modal alignment layers and a gated fusion block — representative of the general discriminative and generative multimodal learning frameworks described throughout this survey.

### Meticulous Low-Level Technical Explanation

The following describes, in granular detail, how heterogeneous data streams flow through the complete system pipeline without any token suppression or information loss:

---

#### Stage 1: Modality-Specific Input Preprocessing

Each input modality undergoes preprocessing into its canonical representation format:

- **Visual stream (Images/Video frames):** Raw pixel arrays of shape `(H × W × C)` for images, or `(T × H × W × C)` for video. Preprocessed via normalization (mean subtraction, standard deviation division), followed optionally by spatial resizing to align with backbone input requirements.

- **Audio stream (Music/Speech/Ambient Sound):** Raw waveform converted to a **mel-spectrogram** representation of shape `(F × T)` where `F` is the number of mel frequency bins and `T` is the number of time steps. For music generation tasks, the audio may alternatively be represented in discrete VQ token space.

- **Text stream (Captions/Dialogue/Q&A):** Raw text tokenized into sub-word units (BPE or WordPiece tokenization) and embedded into dense vector representations via a learned embedding matrix `E ∈ ℝ^(V × d)` where `V` is vocabulary size and `d` is embedding dimensionality.

---

#### Stage 2: Modality-Specific Encoders (Separate Processing Branch)

**Visual Encoder:**
- For images: A CNN backbone (ResNet, VGG) or Vision Transformer (ViT / Masked Autoencoder) encodes the input image into a feature map or sequence of patch embeddings `f_v ∈ ℝ^(N_v × d_v)` where `N_v` is the number of spatial regions or patches.
- For videos: A 3D CNN (e.g., I3D) or video transformer extends this to produce spatio-temporal feature sequences `f_v ∈ ℝ^(T × N_v × d_v)`.

**Audio Encoder:**
- CNN-based audio feature extraction on mel-spectrogram `f_a ∈ ℝ^(N_a × d_a)`.
- Alternatively, for speech: transformer-based encoding aligned with language tokens.
- For music (VQ-based): A VQ-VAE encoder maps audio to discrete codebook indices.

**Text Encoder:**
- RNN (LSTM) for sequential word processing with memory state `h_t = LSTM(x_t, h_{t-1})`, producing a final hidden state `f_t ∈ ℝ^(d_t)` or sequence of hidden states.
- Alternatively, Transformer (BERT, GPT) encoder producing contextual embeddings for each token: `f_t ∈ ℝ^(N_t × d_t)`.

---

#### Stage 3: Cross-Modal Alignment Layer

The alignment layer is the critical bridge between independently encoded modality representations. Two canonical approaches:

**Contrastive Alignment (CLIP-style):**
- Modality-specific projections: `z_v = W_v · f_v ∈ ℝ^d`, `z_t = W_t · f_t ∈ ℝ^d` map each modality into a shared embedding space of dimension `d`.
- Similarity matrix `S = z_v · z_t^T ∈ ℝ^(B × B)` computed across a batch of size `B`.
- InfoNCE/NCE contrastive loss maximizes similarity for positive pairs (matched image-text) and minimizes for negative pairs (mismatched within batch).

**Attention-Based Cross-Modal Fusion:**
- Cross-attention from one modality to another: `Attention(Q_v, K_a, V_a) = softmax(Q_v · K_a^T / √d_k) · V_a`
- Queries from the visual stream attend over keys and values from the audio or text stream, producing attended representations that incorporate cross-modal context.
- Multi-head attention [Vaswani et al. 2017] runs `h` parallel attention heads, each operating in a lower-dimensional subspace: `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O`

---

#### Stage 4: Gated Fusion Block

The gated fusion block combines aligned representations from multiple modalities with learned, dynamic weighting — critically **without suppressing tokens from any stream**:

**Gating mechanism:**
```
g = σ(W_g · [f_v ; f_a ; f_t] + b_g)
f_fused = g ⊙ f_v + (1 - g) ⊙ f_a + additional_modality_terms
```
where `σ` is the sigmoid function, `⊙` is element-wise multiplication, and `W_g` is a learned gate weight matrix. The gating parameters `g ∈ [0, 1]^d` are **input-dependent** — adapting dynamically to each sample to selectively emphasize the most informative modality at each position.

**Without token suppression:** All token representations from each modality encoder are concatenated or aggregated before the gating operation, ensuring no modality token is zeroed out or masked at the fusion stage. This is critical for tasks where the relevant information may appear in any modality — ambient sound may carry temporal event cues absent from the visual stream, or text may resolve ambiguities in the visual scene.

---

#### Stage 5: Task-Specific Decoder Head

The fused representation `f_fused` is passed to a task-specific output head:

- **Discriminative tasks (classification, localization):** Linear classifier `y = softmax(W_c · f_fused + b_c)` for event classification; temporal regression head for localization tasks.
- **Generative tasks (caption, image, audio synthesis):** Autoregressive decoder (LSTM or Transformer decoder) generating output sequences token-by-token conditioned on `f_fused`; or a convolutional decoder for pixel-space image synthesis; or a vocoder for audio waveform synthesis.
- **Retrieval tasks:** Cosine similarity `sim(z_query, z_gallery) = (z_query · z_gallery^T) / (||z_query|| · ||z_gallery||)` in the shared embedding space.

---

# Appendix A: Multimodal Datasets — Complete Catalogue

The following table summarizes all multimodal datasets discussed in the survey. R = Real-World, S = Synthetic.

| Dataset | Nature | Vision | Audio | Text | Others | Typical Applications |
|---|---|---|---|---|---|---|
| MSCOCO [Lin et al. 2014] | R | image | — | caption | body keypoints | image captioning / text-to-image synthesis |
| CC (Conceptual Captions) [Sharma et al. 2018; Changpinyo et al. 2021] | R | image | — | caption | — | image-text pretraining |
| LAION-400M / LAION-5B [Schuhmann et al. 2021; 2022] | R | image | — | caption | — | image-text pretraining |
| CUB200 [Wah et al. 2011] | R | image | — | caption | — | text-to-image synthesis (fine-grained) |
| VisDial [Das et al. 2017] | R | image | — | dialog | — | visual dialog |
| GuessWhat?! [De Vries et al. 2017] | R | image | — | dialog | — | visual dialog |
| VQA / VQA 2.0 [Antol et al. 2015; Goyal et al. 2017] | R | images | — | Q&A | — | Visual Question Answering |
| OK-VQA [Marino et al. 2019] | R | images | — | dialog | — | Knowledge-required VQA |
| NExT-QA [Xiao et al. 2021] | R | video | — | dialog | — | Causal/temporal VQA |
| MSVD [Chen & Dolan 2011] | R | video | — | caption | — | video captioning |
| MSR-VTT [Xu et al. 2016] | R | video | — | caption | — | video captioning |
| Vatex [Wang et al. 2019] | R | video | — | caption | — | multilingual video captioning |
| ActivityNet Captions [Krishna et al. 2017; Zhou et al. 2019] | R | video | — | caption | — | video captioning / grounding |
| Visual Genome [Krishna et al. 2017] | R | images | — | caption | graph | VQA / captioning / grounding |
| WIT [Srinivasan et al. 2021] | R | images | — | multilingual texts | — | multimodal retrieval |
| Crema-d [Cao et al. 2014] | R | video | vocal audio | — | — | emotion perception |
| LipRead [Chung & Zisserman 2017] | R | video | vocal audio | — | — | speech recognition |
| ASL Alphabet [Nagaraj 2018] | R | images | — | sign language | — | sign language recognition / generation |
| How2sign [Duarte et al. 2021] | R | videos | — | sign language | — | sign language recognition / generation |
| AVSD [Alamri et al. 2019; Hori et al. 2019] | R | video | ambient | dialog | — | video captioning / visual dialog |
| AVE [Tian et al. 2018] | R | video | ambient | — | — | audio-visual event localization |
| EPIC-SOUNDS [Huh et al. 2023] | R | videos | ambient | — | — | audio-visual action recognition |
| AIST [Tsuchida et al. 2019] | R | video | music | — | — | dance-music |
| AIST++ [Li et al. 2021] | R | video | music | — | 2D/3D/SMPL | dance-music |
| TikTok Dance-Music [Zhu et al. 2022] | R | video | music | — | 2D | dance-music |
| MUSIC [Zhao et al. 2018; 2019] | R | video | music | — | — | video-music |
| Flickr Audio Caption Corpus [Harwath & Glass 2015] | R | image | speech | caption | — | sanity check for multimodal learning |
| LLP (Look, Listen, Parse) [Tian et al. 2020] | R | video | ambient | — | — | audio-visual video parsing |
| CLEVR [Johnson et al. 2017] | S | image | — | dialog | scene graph | visual question answering |
| MUGEN [Hayes et al. 2022] | S | video | music | caption | — | video-music-text |
| ModelNet [Wu et al. 2015] | S | image | — | — | point clouds/meshes | 3D tasks |
| ShapeNet [Chang et al. 2015] | S | image | — | — | point clouds/meshes | 3D tasks |

### Dataset Detailed Descriptions

**MSCOCO** [Lin et al. 2014 ECCV]: Large-scale benchmark widely used for object detection and segmentation. Includes 330k images, over 200k labeled. For multimodal: includes textual captions per image → applicable to image captioning and text-to-image synthesis.

**Conceptual Captions (CC)** [Changpinyo et al. 2021 CVPR; Sharma et al. 2018 ACL]: Image-caption pairs extracted from the Internet. Training split: over 3M images, token vocabulary size of 51k, average caption length 10.3 words. Validation and testing splits: over 15k images. Often used for large-scale model pretraining. (Note: CLIP [Radford et al. 2021] does not release its training dataset.)

**LAION** [Schuhmann et al. 2021 arXiv; 2022 arXiv]: Open large-scale datasets for vision-language representation learning and generation. Two versions:
- **LAION-400M:** 400 million image-text pairs (used in CLIP training).
- **LAION-5B:** Larger collection of filtered image-text pairs (used for GLIDE [Nichol et al. 2021] and Stable Diffusion [Rombach et al. 2022]).

**CUB200** [Wah et al. 2011]: 11,788 images of 200 bird species (5,994 train / 5,794 test). Textual descriptions per bird image annotated by Reed et al. [2016]. Focused on **very specific and fine-grained** visual category.

**VisDial** [Das et al. 2017 CVPR]: 120k MSCOCO images + dialog annotations. Each image has 10 Q&A pairs → 1.2M rounds of dialog interactions. Benchmark for visual dialog.

**GuessWhat?!** [De Vries et al. 2017 CVPR]: 800k visual Q&A pairs on 66k images. Collected from a cooperative two-player game: one player (oracle) is assigned a random object; other player (questioner) locates the object via dialog. Used for visual dialog task.

**VQA / VQA 2.0** [Antol et al. 2015 ICCV; Goyal et al. 2017 CVPR]: Real-world images with open-ended questions. VQA 2.0 enriches with 10 ground truth + 3 plausible answers for 265,016 images, providing more balanced dataset for studying VQA task.

**OK-VQA** [Marino et al. 2019 CVPR]: "Outside Knowledge" VQA. Answers must be provided based on both image content and **external knowledge** — designed for the scenario where answers cannot be inferred purely from given images.

**NExT-QA** [Xiao et al. 2021 CVPR]: Video-text dataset with **causal and temporal action reasoning** in video question answering. 5,440 videos (average length 44s). 10 Q&A pairs per video → 52k manually labeled pairs. Q&A further grouped into causal, temporal, or descriptive clusters.

**MSVD** [Chen & Dolan 2011 ACL]: ~120k English descriptions for over 2,000 video clips. Initially for machine translation; widely used for video captioning.

**MSR-VTT** [Xu et al. 2016 CVPR]: 10,000 web video clips, 41.2 hours, 2000k clip-sentence pairs. Each clip annotated with 20 natural sentences. Based on 257 popular queries across 20 representative categories. Adopted in video captioning.

**Vatex** [Wang et al. 2019 ICCV]: 41,250 videos, 825,000 captions in English and Chinese, over 206,000 English-Chinese translation pairs. Proposed for multilingual video captioning.

**ActivityNet Captions** [Krishna et al. 2017 ICCV; Zhou et al. 2019 CVPR]: 20k untrimmed YouTube videos with 100k caption annotations. Average video length 120s; over 3 recognized activities per video; average caption 13.5 words. For video captioning and grounding.

**Visual Genome** [Krishna et al. 2017 IJCV]: 108k images annotated with objects, attributes, pairwise relationships as graphs. 5.4M region descriptions, 1.7M visual Q&A. Adopted for VQA, captioning, and grounding.

**WIT** [Srinivasan et al. 2021 SIGIR]: Wikipedia-based image text dataset. 37.5M image-text examples, 11.5M unique images across 108 languages. Proposed mainly for multimodal retrieval.

**Crema-d** [Cao et al. 2014 IEEE]: Facial and vocal emotional expressions in sentences spoken with different emotional states (happy, sad, anger). Designed for multimodal emotion expression and perception.

**LipRead** [Chung & Zisserman 2017 ACCV]: Words/speech recognition from talking faces. Videos from TV broadcasts with over 1 million word instance annotations.

**ASL Alphabet** [Nagaraj 2018]: Images of American Sign Language alphabets. 29 classes (26 alphabets A–Z + SPACE, DELETE, NOTHING). Designed for sign language recognition and generation.

**How2sign** [Duarte et al. 2021 CVPR]: Multimodal and multiview continuous ASL dataset. Over 80 hours of sign language videos. Corresponding modalities: speech, English transcripts, depth.

**AVSD (Audio Visual Scene Aware Dialog)** [Alamri et al. 2019 CVPR; Hori et al. 2019 ICASSP]: Vision, audio, and text data. Visual: videos from Charades dataset [Sigurdsson et al. 2016 ECCV]. Text: caption + dialog annotations. Ten Q&A pairs per dialog per video. Applicable for video captioning [Zhu et al. 2020; 2021] and VQA [Schwartz et al. 2019].

**AVE (Audio Visual Event)** [Tian et al. 2018 ECCV]: 4,143 unconstrained 10-second videos covering 28 event categories (dog barking, frying food, playing guitar, etc.). Most videos contain visual events spanning the full 10 seconds with corresponding ambient sound. Proposed for audio-visual event localization.

**EPIC-SOUNDS** [Huh et al. 2023 ICASSP]: Built on EPIC-KITCHENS-100 [Damen et al. 2018 ECCV]. 75.9k segments of audible events and actions across 44 classes. For audio-visual localization and recognition.

**AIST** [Tsuchida et al. 2019 ISMIR]: Professional dance videos with paired dance music. 13,940 videos, 1,618 dances, 60 music pieces. Total: ~118.1 hours. 10 different genres (break, pop, house, street jazz, etc.). Performed by professional dancers in studio from 9 camera poses.

**AIST++** [Li et al. 2021 ICCV]: Subset of AIST with refined dance motion annotations: SMPL [Loper et al. 2015] representations, 3D human body keypoints, 2D human keypoints. Proposed for **music-to-dance generation**.

**TikTok Dance-Music** [Zhu et al. 2022 ECCV]: 445 video clips (average 12.3 seconds, 85 songs) from TikTok. Contains 2D human skeleton data. **"In the wild"** counterpart to AIST++ — more challenging due to unconstrained capture conditions.

**MUSIC** [Zhao et al. 2018 ECCV; Zhao et al. 2019 ICCV]: 685 untrimmed videos of musical solos and duets, 11 instrumental types (guitar, cello, violin, etc.). Initially proposed for visual-audio grounding (spatial sound localization).

**Flickr Audio Caption Corpus (FACC)** [Harwath & Glass 2015]: 8,000 natural images × 5 textual descriptions = 40,000 captions with speech audio. Small dataset with vision, audio, and text annotations. Used for preliminary experiments and sanity checks.

**LLP (Look, Listen, Parse)** [Tian et al. 2020 ECCV]: Subset of AudioSet [Gemmeke et al. 2017 ICASSP]. 11,849 YouTube video clips, 25 event classes. Each 10-second video with ambient sound characterizing at least one event. 6,626 event annotations: 4,131 audio + 2,495 visual, with second-wise temporal boundaries.

**CLEVR** [Johnson et al. 2017 CVPR]: Diagnostic synthetic dataset for visual reasoning and compositional language parsing. Geometric shapes (cubes, cylinders, spheres) in different colors and materials. 70k/15k/15k images (train/validation/test) + Q&A annotations.

**MUGEN** [Hayes et al. 2022 arXiv]: Synthetic dataset from CoinRun open-source platform game [Cobbe et al. 2019 ICML]. 375k video clips of 3.2 seconds with paired music sequences and human-annotated text descriptions.

**ModelNet** [Wu et al. 2015 CVPR]: Synthetic 3D dataset with multiple 3D annotations (point clouds, CAD-generated meshes). Two variants: ModelNet10 and ModelNet40 (different numbers of object categories). Widely used for 3D multimodal tasks.

**ShapeNet** [Chang et al. 2015 arXiv]: Large-scale 3D CAD model repository, 16 common object classes, over 300M individual meshes, 220,000 classified into 3,135 classes annotated using WordNet hyponym-hypernym relationships.

---

# Appendix B: Evaluation for Synthesized Data — Complete Metrics Reference

## B.1 Visual Data Evaluation

### General Quality Metrics

**IS (Inception Score)** [Salimans et al. 2016 NeurIPS]:
Measures general **reality and diversity** for generated images using a pre-trained Inception v3 [Szegedy et al. 2016 CVPR] image classification model. Based on two premises:
1. A synthesized image `x` should be assigned to a class discriminatively → low entropy of conditional label distribution `p(y|x)`.
2. Diversity: marginal probabilities across all labels should be widely spread → high entropy of marginal `p(y)`.
IS combines both criteria. **Higher IS = better quality**.

**FID (Fréchet Inception Distance)** [Heusel et al. 2017 NeurIPS]:
Calculates the distance on feature vector levels. Computes image feature vectors using Inception models [Szegedy et al. 2016] from both original and generated images. Measures the Fréchet distance between two Gaussian distributions fit to real vs. generated feature vectors. **Lower FID = better quality** (higher similarity to real images).

**FVD (Fréchet Video Distance)** [Unterthiner et al. 2019]:
Designed to measure the **fidelity of generated videos**. Analogous to FID but extended to the video domain, using 3D convolutional features.

**Subjective Evaluations:**
Human user evaluations conducted case-by-case. Example: DALL·E 2 [Ramesh et al. 2022] performs extensive human evaluations in terms of photorealism, caption similarity, and sample diversity.

### Cross-Modal Metrics

**ClipScore** [Hessel et al. 2021 EMNLP]:
An **objective metric** to measure the relevance between given text description and synthesized image. Works by computing similarity on the feature level of a text-image pair via the pre-trained CLIP model [Radford et al. 2021]. Used in [Tang et al. 2022; Zhu et al. 2023 ICLR (CDCD)].

**Classification Accuracy** [Li et al. 2019 CVPR]:
For class-conditioned image generation: pre-trained classifier-based scores measuring whether synthesized image well corresponds to the given class label.

## B.2 Audio Data Evaluation

### Music Audio

**SNR (Signal-to-Noise Ratio)** [Plapous et al. 2006]: Quantifies the strength of desired audio signal relative to background noise. General quality objective metric.

**MOS (Mean Opinion Score)**: Subjective human evaluation via crafted instructions. Used in [Zhu et al. 2022 ECCV; Kumar et al. 2019 NeurIPS].

**Beat Coverage and Beat Hit Scores** [Zhu et al. 2022 ECCV; Zhu et al. 2023 ICLR (CDCD)]: Objective metrics measuring consistency of musical rhythms with respect to given video input for human body dance motions.

**Genre Accuracy (Classifier-based retrieval accuracy)** [Zhu et al. 2022 ECCV]: Considers diversity of generated music genres.

**Pitch Class Histogram Entropy** [Wu & Yang 2020 arXiv]: Evaluates music in terms of **tonality** — distribution of pitch classes across the chromatic scale.

**Grooving Pattern Similarity** [Wu & Yang 2020 arXiv]: Measures **rhythmicity** of generated music.

**Structureness Indicator** [Wu & Yang 2020 arXiv]: Evaluates the **repetitive structure** of music audio — a key quality of well-structured musical compositions.

### Speech Audio

**STOI (Short-Time Objective Intelligibility)** [Taal et al. 2010 ICASSP]: Measures quality of **speech audio intelligibility**.

**ESTOI (Extended STOI)** [Jensen & Taal 2016 IEEE/ACM]: Extended version of STOI for improved intelligibility measurement under modulated noise maskers.

**PESQ (Perceptual Evaluation of Speech Quality)** [Rix et al. 2001 ICASSP]: Indicator for perceptual quality of speech audio. Standardized method for telephone network and codec quality assessment.

**WER (Word Error Rate)**: Counts correct word predictions in comparison to the ground truth. Used for evaluating speech-to-text accuracy in ASR systems.

**MOS (Mean Opinion Score) for Speech** [Tan et al. 2022 arXiv (NaturalSpeech)]: Human-based test for assessing quality in recent speech synthesis studies.

### Ambient Audio

Existing works largely rely on **human-based subjective evaluations** with specific assessing criteria and instructions for different aspects (sound-video alignment, real-or-fake determination). Few works also report **cross-entropy loss** as quantitative metric scores [Zhou et al. 2018 CVPR].

## B.3 Textual Data Evaluation

Evaluation metrics for textual data are well-developed in the NLP field and widely adopted for generative tasks.

**BLEU (BiLingual Evaluation Understudy)** [Papineni et al. 2002 ACL]:
Initially designed for evaluating machine-translated texts; believed to have high correlation with human judgments. Computes **n-gram precision** — counting matches for n-gram words in generated vs. ground truth sentences. Range: 0 to 1. **Larger value = higher similarity to reference**.

**METEOR (Metric for Evaluation of Translation with Explicit ORdering)** [Banerjee & Lavie 2005 ACL Workshop]:
Additionally considers **alignment between evaluated and reference sentences** in terms of harmonic mean of unigram precision and recall.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** [Lin 2004]:
Measures similarity between translated textual summary and reference summary via **n-gram matches**. ROUGE value close to 0 = poor similarity; value close to 1 = high similarity.

**CIDEr (Consensus-based Image Description Evaluation)** [Vedantam et al. 2015 CVPR]:
Specifically proposed for measuring quality of textual description with regard to images. Compares generated sentences with a **set of ground truth sentences written by humans**, showing higher agreement with consensus of human evaluators.

**SPICE (Semantic Propositional Image Caption Evaluation)** [Anderson et al. 2016 ECCV]:
Specifically designed for caption evaluations. Takes **semantic propositional content** into account; calculates score defined over the scene graph. Value closer to 1 implies better quality.

**CLIPScore** [Hessel et al. 2021 EMNLP]:
Proposed as a **reference-free metric** for text and image related tasks using distance computation from the feature space of the pre-trained CLIP model [Radford et al. 2021]. Applicable for both visual and textual evaluation in cross-modal generation.

### Summary Table of All Evaluation Metrics

| Metric | Applicable Data | Gen./Cross I-O | Obj./Sub. | Measures |
|---|---|---|---|---|
| IS [Salimans et al. 2016] | Image | Gen. | Obj. | quality & diversity |
| FID [Heusel et al. 2017] | Image | Gen. | Obj. | quality |
| FVD [Unterthiner et al. 2019] | Video | Gen. | Obj. | quality |
| ClipScore [Hessel et al. 2021] | Image & Texts | Cross I-O | Obj. | text-image correspondence |
| Classification Acc. [Li et al. 2019] | Image & Video | Cross I-O | Obj. | label correspondence |
| SNR [Plapous et al. 2006] | General audio | Gen. | Obj. | signal-to-noise ratio |
| MOS | General audio | Gen. & Cross I-O | Sub. | crafted instruction quality |
| Beat Scores [Zhu et al. 2022] | Music | Cross I-O | Obj. | beat correspondence |
| Genre Accuracy [Zhu et al. 2022] | Music | Cross I-O | Obj. | genre diversity |
| Pitch Class Histogram Entropy [Wu & Yang 2020] | Music | Gen. | Obj. | music tonality |
| Grooving Pattern Similarity [Wu & Yang 2020] | Music | Gen. | Obj. | music rhythm |
| Structureness Indicator [Wu & Yang 2020] | Music | Gen. | Obj. | music repetitive structure |
| STOI [Taal et al. 2010] | Speech | Gen. | Obj. | speech intelligibility |
| ESTOI [Jensen & Taal 2016] | Speech | Gen. | Obj. | speech intelligibility |
| PESQ [Rix et al. 2001] | Speech | Gen. | Obj. | perceptual speech quality |
| WER | Speech | Cross I-O | Obj. | speech-word accuracy |
| BLEU [Papineni et al. 2002] | Text | Gen. | Obj. | text similarity |
| METEOR [Banerjee & Lavie 2005] | Text | Gen. | Obj. | text similarity |
| ROUGE [Lin 2004] | Text | Gen. | Obj. | text similarity |
| CIDEr [Vedantam et al. 2015] | Text | Cross I-O | Obj. | text similarity (image-specific) |
| SPICE [Anderson et al. 2016] | Text | Cross I-O | Obj. | text-image semantic similarity |

---

# Section 7: Conclusions

This survey presents multimodal machine learning from the **unique perspective of data characteristics**. The analysis proceeds through:

1. **Data Modality Analysis:** Intrinsic natures of vision, audio, and text — analyzing characteristics, commonalities, and unique attributes.

2. **Multimodal Representation Learning:** Current literature categorized by learning settings (supervised and non-supervised), with the primary shift from supervised methods with annotated data toward large-scale pretraining on web-scale paired data.

3. **Discriminative Applications:** Concrete task applications from both discriminative natures, structured into sub-classes with specific data combinations in Vision+X form. Includes a revisit *in light of data* — bridging existing technique designs and their connections to data nature from different modalities.

4. **Generative Applications:** Introduction to popular generative backbone models (VAE, GAN, DPM) before detailed task explanations.

5. **Challenges and Future Directions:** Discussions on existing challenges and promising future directions for the multimodal learning domain.

The survey's central thesis — **that understanding the intrinsic nature of each data modality is essential for principled multimodal system design** — manifests in two complementary findings:
- **Exploiting uniqueness:** Emphasizing and exploiting unique characteristics of specific data modalities contributes to solving concrete application problems associated with those modalities.
- **Leveraging commonalities:** Recognizing commonalities among different modalities enables constructing a more unified and collaborative framework mirroring the capabilities of a real human intelligence system.

---

*Document compiled from: Zhu, Y., Wu, Y., Sebe, N., & Yan, Y. (2022/2024). Vision + X: A Survey on Multimodal Learning in the Light of Data. Journal of LaTeX Class Files / arXiv:2210.02884v2.*
