# Deep Multimodal Learning with Missing Modality (MLMM)
## A Comprehensive Reference Manual

> **Source:** Wu et al. (2026). *Deep Multimodal Learning with Missing Modality: A Survey.*
> Published in *Transactions on Machine Learning Research* (02/2026).
> arXiv:2409.07825v4 [cs.CV] 4 Feb 2026.
> Reviewed on OpenReview: https://openreview.net/forum?id=tc7RFcx4hT

**Authors:**
- Renjie Wu — The Australian National University (renjie.wu@anu.edu.au)
- Hu Wang — Mohamed bin Zayed University of Artificial Intelligence (hu.wang@mbzuai.ac.ae)
- Hsiang-Ting Chen — Adelaide University (tim.chen@adelaide.edu.au)
- Gustavo Carneiro — The University of Surrey (g.carneiro@surrey.ac.uk)

---

## Table of Contents

1. [Project Foundations: Architectural Overview & Theoretical Scope](#1-project-foundations-architectural-overview--theoretical-scope)
2. [Theoretical Framework: Modality, Medium, and the Missing Modality Problem](#2-theoretical-framework-modality-medium-and-the-missing-modality-problem)
3. [Methodology Taxonomy: An Overview](#3-methodology-taxonomy-an-overview)
4. [Methodologies in Data Processing Aspect](#4-methodologies-in-data-processing-aspect)
   - 4.1 [Modality Imputation](#41-modality-imputation)
   - 4.2 [Representation-Focused Models](#42-representation-focused-models)
5. [Methodologies in Strategy Design Aspect](#5-methodologies-in-strategy-design-aspect)
   - 5.1 [Architecture-Focused Models](#51-architecture-focused-models)
   - 5.2 [Model Combinations](#52-model-combinations)
6. [Methodology Discussion & Technical Evolution](#6-methodology-discussion--technical-evolution)
7. [Applications and Datasets](#7-applications-and-datasets)
8. [Open Issues and Future Research Directions](#8-open-issues-and-future-research-directions)
9. [System Architecture & Visual Pipeline](#9-system-architecture--visual-pipeline)
10. [Conclusion](#10-conclusion)

---

# 1. Project Foundations: Architectural Overview & Theoretical Scope

## 1.1 The Multimodal Learning Paradigm

Multimodal learning has become a crucial field in Artificial Intelligence (AI). It focuses on **jointly analyzing various data modalities**, including visual, textual, auditory, and sensory information. This approach mirrors the human capacity to combine multiple senses for better understanding and interaction with the environment.

Modern multimodal models leverage the **robust generalization capabilities of deep learning** to uncover complex patterns and relationships that single-modal systems might not detect. This capability is advancing work across multiple domains (Bayoudh et al., 2022; Hu et al., 2023; Streli et al., 2023).

Recent surveys show that multimodal learning significantly boosts performance and enables more advanced AI applications (Baltrušaitis et al., 2018; Zhao et al., 2024; Wu et al., 2024a).

## 1.2 The Core Problem: Missing Modalities in Real-World Systems

Real-world multimodal systems **frequently encounter scenarios where certain data modalities are missing or incomplete**. Such cases arise from:

- **Sensor malfunctions** — hardware failure during data acquisition
- **Hardware limitations** — physical constraints of collection devices
- **Privacy concerns** — certain data types legally or ethically restricted
- **Environmental interference** — noise, occlusion, lighting conditions
- **Data transmission failures** — packet loss in streaming pipelines

Interest in addressing this challenge has grown considerably in recent years, as reflected in the publication trends from 2014 through 2025 (see Figure 1a of the survey): publications grew from fewer than 5 per year in 2014 to over 55 per year by 2025.

### 1.2.1 Full-Modality vs. Missing-Modality Samples

In a **three-modality setting**, data samples can be classified as:

- **Full-Modality Samples** — samples containing all three modalities (Mod 1, Mod 2, Mod 3) fully present
- **Missing-Modality Samples** — samples in which one or more modalities are absent; for example, a sample might have Mod 1 and Mod 3 present but Mod 2 absent, or Mod 1 only, or Mod 3 only

![Full-Modality vs Missing-Modality Samples](IMG_20260612_151314.jpg)

*Figure 1b (survey): Description of a three-modality scenario with full- and missing-modality samples. "Modality" is abbreviated as "Mod" in the survey figures. Dashed boxes with fading colors represent missing modalities/modules. The diagram visually contrasts a sequence of complete triplets (full-modality) against a sequence where any of the three modality slots may be empty (missing-modality).*

Missing modalities can occur **at any stage**, from data collection to deployment, and can substantially degrade model performance.

## 1.3 Historical Precedents and Motivating Examples

### Early Affective Computing

Early works in affective computing (Cohen et al., 2004; Sebe et al., 2005) reported that when collecting facial and audio data, some image or audio samples were not useful due to:
- Excessive microphone noise
- Camera obstructions blocking facial capture

This forced researchers to propose audio-visual models that could handle missing modalities to recognize human emotional states.

### Medical AI

In the field of medical AI, **privacy concerns** and the challenges of obtaining certain data modalities during surgeries or invasive treatments often lead to missing modalities in multimodal datasets (Zhang et al., 2023b; Li et al., 2025b).

### Space Exploration

NASA's Ingenuity Mars helicopter (Donaldson, 2024) faced a missing modality challenge when its inclinometer failed due to extreme temperature cycles on Mars. To address this issue, NASA applied a software patch that modified the initialization of navigation algorithms (Team, 2022).

### Structural and Quantitative Heterogeneity

Beyond domain-specific cases, **structural or quantitative heterogeneity** across sensor types, versions, or brands can also yield inconsistent modality availability. Combined with the unpredictability of real-world scenarios and the diversity of data sources, these factors make robustness to missing modalities an essential requirement for multimodal systems.

## 1.4 Survey Scope and Contributions

This survey presents the **first comprehensive review of Multimodal Learning with Missing Modality (MLMM)**, with a focus on deep learning approaches. Specific contributions:

**(1)** A comprehensive survey of MLMM methods across diverse domains, accompanied by an extensive compilation of relevant datasets, highlighting the versatility of MLMM in addressing real-world challenges.

**(2)** A novel, fine-grained taxonomy of MLMM methodologies, with a **multi-faceted categorization framework** based on multi-modal integration stages and missing modality recovery strategies.

**(3)** An in-depth analysis of current MLMM approaches, their challenges, and future research directions, contextualized within the proposed taxonomical framework.

### Paper Collection Methodology

Papers were sourced primarily from **Google Scholar** and major conferences/journals in:
- AI, Machine Learning, Computer Vision
- Natural Language Processing, Audio Signal Processing
- Data Mining, Multimedia, Medical Imaging
- Remote Sensing and Pervasive Computing

Collected papers are from, but not limited to, **top-tier conferences**: AAAI, IJCAI, NeurIPS, ICLR, ICML, CVPR, ICCV, ECCV, ACL, EMNLP, KDD, ACM MM, MICCAI, ICASSP.

And **journals**: TPAMI, TIP, TMI, TMM, JMLR, TMLR.

A total of **354 significant papers** were compiled from the period spanning **2012 to August 2025**.

**Search keywords used:** "incomplete," "missing," "partial," "absent," "imperfect," combined with terms like "multimodal learning," "deep learning," "representation learning," "multi-view learning," and "neural networks."

---

# 2. Theoretical Framework: Modality, Medium, and the Missing Modality Problem

## 2.1 Definition of the Missing Modality Problem

In this survey, the challenge of handling incomplete inputs in multimodal learning is referred to as the **missing modality problem**. Solutions to this challenge are termed **Multimodal Learning with Missing Modality (MLMM)**.

### Critical Distinction: MLMM vs. Conventional Multimodal Learning

| Property | Conventional Multimodal Learning | MLMM |
|---|---|---|
| Modality availability | All N modalities consistently available | One or more modalities absent during training or testing |
| Assumption | Full-modality setting assumed | Must operate under incomplete modality conditions |
| Goal | Jointly analyze available modalities | Robustly exploit available modalities while maintaining performance comparable to full-modality systems |

**The central goal of MLMM** is to robustly exploit the available modalities while maintaining performance comparable to systems trained with complete modality information.

## 2.2 Application Domains of MLMM

This survey reviews recent advances in MLMM and its applications across diverse domains:

- **Information retrieval** (Malitesta et al., 2024; Pipoli et al., 2025)
- **Remote sensing** (Wei et al., 2023)
- **Pervasive Computing** (Nweke et al., 2018; Van Der Donckt et al., 2024; Chen et al., 2023a)
- **Robotic vision** (Gunasekar et al., 2020; Park et al., 2025)
- **(Bio-)medical domain** (Zhou et al., 2023; Shah et al., 2023; Azad et al., 2022; Zhu et al., 2025)
- **Sentiment analysis** (Wagner et al., 2011; Wang et al., 2025b)
- **Multi-view clustering** (Chao et al., 2021)

---

# 3. Methodology Taxonomy: An Overview

The survey reviews current deep MLMM methods from **two key aspects**: data processing and strategy design. These are based on the contributions of existing work.

![MLMM Taxonomy Diagram](IMG_20260612_151314.jpg)

*Figure 2 (survey): The taxonomy of deep multimodal learning with missing modality methods. Existing methods are categorized into two aspects — Data Processing and Strategy Design — each further divided into subcategories. The diagram presents a hierarchical breakdown: "Deep Multimodal Learning with Missing Modality" at the root, splitting into "Data Processing Aspect" (left branch: Modality Imputation → Modality Composition, Modality Generation; Representation-Focused → Coordinated-Representation, Representation Composition, Representation Generation) and "Strategy Design Aspect" (right branch: Architecture-Focused → Attention-Based, Distillation-Based, Graph Learning-Based, MLLMs; Model Combinations → Ensemble, Dedicated, Discrete Scheduler). "MLLMs" stands for multimodal large language models.*

## 3.1 Data Processing Aspect

Methods that focus on exploring the data processing aspect of a model or framework can be divided into **Modality Imputation** and **Representation-Focused Models**, depending on whether the model's handling of missing modalities occurs at the modality data level or at the data representation level.

### 3.1.1 Modality Imputation (Data Level)

Operates at the **modality data level** and fills in the missing information by:
- **Compositing** (modality composition methods): Using zero/random values or data copied from similar instances
- **Generating** (modality generation methods): Using generative models to synthesize absent modalities

Core idea: if a missing modality can be accurately imputed, downstream tasks can continue as if "full" modalities were available.

### 3.1.2 Representation-Focused Models (Representation Level)

Address missing modalities at the **representation level**:
- **Coordinated-representation methods**: Impose specific constraints to align representations of available modalities in the same semantic space
- **Representation composition methods**: Combine representations of existing modalities to fill in the gaps
- **Representation generation methods**: Generate the missing-modality representation using available data

## 3.2 Strategy Design Aspect

Methods based on models that can **dynamically adapt** to different missing-modality cases during training and testing through:
- **Flexible adjustments of the model architecture** (internal model architecture adjustment) → Architecture-Focused Models
- **Combination of multiple models** (external model combinations) → Model Combinations

### 3.2.1 Architecture-Focused Models

Address missing modalities by designing **flexible model architectures** that can adapt to varying numbers of available modalities during training or inference. Key techniques:

- **Attention mechanisms** (Ge et al., 2023; Yao et al., 2024; Gong et al., 2023; Mordacq et al., 2024; Radosavovic et al., 2024; Jang et al., 2024; Qiu et al., 2023): Dynamically adjust modality fusion and processing
- **Knowledge distillation** (Wang et al., 2020a; Saha et al., 2024; Zhang et al., 2023b; Chen et al., 2024b; Wang et al., 2023a; Shi et al., 2024; Wang et al., 2023c): Transfer knowledge from full-modality models to those operating with incomplete data
- **Graph learning-based methods** (Zhao et al., 2022; Zhang et al., 2022; Lian et al., 2023; Malitesta et al., 2024): Exploit natural relationships between modalities using graph structures
- **MLLMs** (Zhan et al., 2024; Wu et al., 2023b; Li et al., 2023a): Use long-context handling and feature processing to accept and process representations from any number of modalities

### 3.2.2 Model Combinations

Tackle missing modality problems by employing strategies that leverage **multiple models or specialized training techniques**:

- **Dedicated training strategies** (Zeng et al., 2022; Chen et al., 2022; Xue & Marculescu, 2023): Tailored for different modality cases
- **Ensemble methods** (Hu et al., 2020; Wang et al., 2021; Lee et al., 2023b): Combine models trained on partial/full modality sets
- **Discrete scheduler methods** (Wu et al., 2023a; Shen et al., 2024; Surís et al., 2023): Incorporate various downstream modules to flexibly process any number of modalities

---

# 4. Methodologies in Data Processing Aspect

## 4.1 Modality Imputation

**Definition:** Modality imputation refers to a technique used by MLMM methods that fills in missing-modality samples or generates missing modalities to complete the dataset with missing modalities by performing various transformations or operations on existing modalities.

Modality imputation methods addressing the missing modality problem at the data modality level can be categorized into **two types**:

**(1) Modality composition methods:** Use zero/random values or data copied from similar instances as the input for the missing modality data. The data produced via these methods to represent the missing data are then composited with the data from the available modalities to form a "full"-modality sample.

**(2) Modality generation methods:** Generate the missing modality data using generative models such as:
- Auto-Encoders (Hinton & Zemel, 1993)
- Generative Adversarial Networks (GANs) (Goodfellow et al., 2014)
- Diffusion Models (Ho et al., 2020)

The generated data is then composited with the data from the available modalities to form a "full"-modality sample.

### 4.1.1 Modality Composition Methods

Modality composition methods are **widely employed for simplicity and effectiveness** to maintain the original dataset size.

#### Zero/Random Values Composition Methods

![Zero/Random Values Composition Method](IMG_20260612_151314.jpg)

*Figure 3 (survey): Zero/Random values composition methods. If modality 2 is missing, this modality is replaced with zero/random values. The diagram shows: [Mod 1] + [Mod 2 → Zero/Random Values] + [Mod 3] → DNN → Representation → Downstream Task. "DNN" in all figures of the survey means different kinds of deep neural networks.*

**Mechanism:** Replace a missing modality with zeros or random values.

**Recent usage (Chen et al., 2020; Sun et al., 2024b; Liu et al., 2023a; Malitesta et al., 2024):** Often used as a **baseline method** for comparison with more sophisticated methods.

**Frame-Zero method** (Parthasarathy & Sundaram, 2020): Proposed to replace missing frames in sequential data such as videos.

**Advantages of zero/random composition methods:**
- Prevalent in typical multimodal learning procedures
- Can be used to balance and integrate information from different modalities when making predictions
- Prevents the model from over-relying on dominant modalities available for each sample
- Enhances robustness by encouraging a more balanced integration of information across all available modalities

#### Retrieval-Based Representation Composition Methods

![Retrieval-Based Modality Composition](IMG_20260612_151314.jpg)

*Figure 4 (survey): Retrieval-based modality composition methods. A sample library is queried (e.g., using KNN or its variants) to find same-category samples that have the required missing modalities. Retrieved samples are processed and then composed with the input missing-modality sample to form a "full"-modality sample. Data flow: Sample Library → [e.g., KNN...] → Retrieval Samples → Process, then fill in → [Mod 1] + [Mod 2 filled] + [Mod 3] → DNN → Representation → Downstream Task.*

**Mechanism:** Replace the missing modality data by copying or averaging data from retrieved samples with the same classification. Some methods randomly select a sample that has the same classification and required missing modality from other samples.

**Limitations:**
- **Not applicable** to pixel-level tasks such as segmentation
- Only suitable for simple tasks (e.g., classification)
- May lead to overfitting of noisy data if mismatched samples are combined
- Cannot solve the missing modality problem during the **testing phase** because they rely on known labels of training data samples
- Retrieval-based methods may introduce duplicated training samples

**Modal-mixup** (Yang et al., 2024): Proposed to complete training datasets by randomly complementing same-category samples with missing modalities. However, cannot solve the missing modality problem during the testing phase.

**Frame-Repeat** (Parthasarathy & Sundaram, 2020): Proposed to make up for missing frames in streaming audio-visual classification tasks (such as audio-visual expression recognition) by using past frames.

#### KNN-Based Retrieval Composition

**Other methods** (Yang et al., 2024; Campos et al., 2015) use **K-Nearest Neighbors (KNN)** or its variants to retrieve the best-matched samples for composition. For matched samples, they:
- Select the sample with the **highest score**, or
- Obtain the **average values** of these samples to supplement the missing modality data

**Experimental findings:** KNN-based methods generally **perform better** than zero/random value methods, and can handle missing modality during testing.

**Drawbacks of KNN retrieval-based methods:**
- **High computational complexity**
- **Sensitivity to imbalanced data**
- **Significant memory overhead**

**Cluster-based approach** (Hussein et al., 2024a): In pervasive computing, proposed to cluster sensor data and pre-learn the best imputation pattern for each cluster combination, enabling fast lookup-based completion of missing modality at runtime.

**Critical Shared Limitation of All Composition Methods:**

All methods above can complete datasets with missing modalities, but they **reduce the diversity of the dataset** because they may introduce duplicated training samples. This is especially problematic with datasets having a high rate of modality missing, where most samples have missing-modality data, as it increases the risk of **overfitting** to certain classes with few full-modality samples.

### 4.1.2 Modality Generation Methods

With deep learning, synthesizing missing modalities has become more effective by leveraging powerful representation learning and generative models that can capture **complex cross-modal relationships**. Current methods for generating missing-modality data are divided into **individual** and **unified** generative methods.

#### Individual Modality Generation Methods

![Individual Modality Generation Methods](IMG_20260612_151314.jpg)

*Figure 5a (survey): Individual modality generation methods. For each modality that may be missing, a separate generator (GEN-2 for modality 2) is trained. Given Mod 1 and Mod 3 as inputs, GEN-2 generates Mod 2. Then [Mod 1] + [Mod 2 generated] + [Mod 3] → DNN → Representation → Downstream Task. Each missing modality requires a dedicated generator.*

**Definition:** Train an **individual generative model for each modality** in case any modality is missing.

**Historical progression:**

- **Early works:** Gaussian processes (Mario Christoudias et al., 2010) or Boltzmann machines (Sohn et al., 2014) to generate missing modalities from available data
- **Deep learning era — Auto-Encoders (AEs):** (Hinton & Zemel, 1993) used for missing modality generation
  - Li et al. (2014): Used 3D-CNN to generate positron emission tomography (PET) data from magnetic resonance imaging (MRI) inputs
  - Chen et al. (2024a): Addressed missing modalities in MRI segmentation by training U-Net models to generate other two modalities from MRI
  - Ma et al. (2021c): Used AEs as one of the baselines to complete datasets by training one AE per modality
  - Zhang et al. (2021): Proposed a U-Net-based Multi-Modality Data Generation module with domain adversarial learning to generate each missing modality by learning domain-invariant features (domain adaptation)
  - DeepLIIF (Ghahremani et al., 2022): Uses ResNet to generate various immunohistochemical modalities (Hematoxylin, mpIF DAPI, mpIF Lap2, mpIF Ki67)
  - HE2RNA (Schmauch et al., 2020): Cross-modal model that predicts gene expression (transcriptomic modality) from H&E whole-slide images (visual modality), generates spatialized gene-expression heatmaps

- **GANs (Generative Adversarial Networks)** (Goodfellow et al., 2014):
  - Significantly improved image generation quality using a generator to create realistic data and a discriminator to distinguish it from real data
  - Arya & Saha (2021): GANs generate missing modalities using latent representations of existing ones in breast cancer prediction
  - Bischke et al. (2018): Used GANs to generate depth data in remote sensing, improving segmentation over RGB-only models
  - Gunasekar et al. (2020): GANs used in robotic object recognition models to generate missing RGB and depth images
  - Ma et al. (2021c): Recent studies show that GANs **outperform AEs** in generating more realistic missing modalities and can lead to better downstream-task model performance
  - SensorGAN (Hussein & Bhat, 2024): Introduces a GAN model that restores absent sensor channels, directly targeting missing-modality recovery in wearable human-activity-recognition pipelines

- **Diffusion Models** (Ho et al., 2020):
  - Wang et al. proposed the **IMDer method** (Wang et al., 2024c): Uses available modalities as conditions to make diffusion models generate missing modalities
  - Experiments showed it can reduce semantic ambiguity between recovered and missing modalities and achieves good generalization performance

#### Unified Original Data Generative Methods

![Unified Modality Generation Methods](IMG_20260612_151314.jpg)

*Figure 5b (survey): Unified modality generation methods. A single generator GEN takes all available modalities as input and generates all modalities simultaneously (including missing ones). Data flow: [Mod 1 available] + [Mod 2 missing] + [Mod 3 available] → GEN → [Mod 1 generated/kept] + [Mod 2 generated] + [Mod 3 generated/kept] → DNN → Representation → Downstream Task. A single unified generative model handles all modality generation jointly.*

**Definition:** Train a **unified model that can generate all modalities simultaneously**.

**Representative models:**

- **Cascade AE** (Tran et al., 2017): Stacks AEs to capture the differences between missing and existing modalities for generating all missing modalities

- **Zhang et al. (2023c):** Attempted to use attention and max-pooling to integrate features of existing modalities, enabling modality-specific decoders to accept available modalities and generate other missing modalities. Demonstrated to be more effective than using max-pooling alone (Chartsias et al., 2017)

**Limitations of all modality generation methods:**
- When facing a scenario where one of the modalities is **severely missing**, training a generator that can produce high-quality missing modalities remains challenging
- The modality generation model **significantly increases storage and computational requirements**
- As the **number of modalities grows**, the complexity of these generative models also increases, further complicating the training process and resource demands

---

## 4.2 Representation-Focused Models

Representation-focused models address the missing modality problem **at the representation level**. They are introduced in the following order:
1. Coordinated-representation-based approaches
2. Representation composition methods
3. Representation generation methods

### 4.2.1 Coordinated-Representation Methods

![Coordinated-Representation Methods](IMG_20260612_151314.jpg)

*Figure 6 (survey): Illustration of coordinated-representation methods. Each modality (Mod 1, Mod 2, Mod 3) is passed through its own DNN encoder (DNN-1, DNN-2, DNN-3) to produce per-modality representations (Representation-1, Representation-2, Representation-3). Constraint Terms (e.g., Deep Canonical-Correlation Analysis or Tensor Rank Regularization) are applied across these representations to make the learned representations semantically consistent. The combined Representation is passed to the Downstream Task. When modality 2 is missing, these methods use constraint terms between the features of different modalities to maintain alignment.*

**Definition:** Coordinated-representation methods focus on introducing **certain constraints between representations of different modalities** to make the learned representations semantically consistent.

**Two categories:**

**Category 1 — Regularization-based:**

- **Tensor Rank Regularization (TRR)** (Liang et al., 2019): Achieves multimodal fusion by taking the outer products of different-modality tensors and taking the sum of all their outer products. To solve the high rank of multimodal representation tensors caused by imperfect modalities in time series, Paul et al. introduced **tensor rank minimization** to try to keep the multimodal representation low rank, thereby alleviating the imperfection of the input

**Category 2 — Correlation-driven:**

- **Deep Canonical Correlation Analysis (CCA)** (Wang et al., 2020b): Maximizes feature associations of available modalities by using Canonical Correlation Coefficient, enabling training on incomplete datasets
- **Maximum likelihood function** (Ma et al., 2021b): Proposed to characterize conditional distributions of full-modality and missing-modality samples during training
- **Hilbert-Schmidt Independence Criterion (HSIC)** (Liu et al., 2021b): Helps models learn how to complete missing modality features by enforcing independence between irrelevant features while maximizing the dependence between relevant ones
- **Additional methods** (Matsuura et al., 2018; Ma et al., 2021a; Li et al., 2022a): Aid models in learning how to complete missing modality features or train on incomplete datasets by learning the similarity or correlation between features

**Drawback of coordinated-representation methods:** They perform well only when **two or three modalities** are used as input (Zhao et al., 2024), and their effectiveness tends to **degrade as the number of modalities increases**. Even when trained on a dataset with missing modalities, some methods still struggle to effectively address **unseen missing modality combinations** or cases during testing.

### 4.2.2 Representation Composition Methods

There are **two types** of representation composition methods:

#### Retrieval-Based Representation Composition Methods

**Mechanism:** Attempt to recover the missing modality representation by retrieving the modality data from existing samples, similarly to the modality composition method in section 3.1.1. They typically:
- Use pre-trained feature extractors to generate features from available samples
- Store them in a **feature pool** (Zhi et al., 2024; Sun et al., 2024b)
- Use **cosine similarity** to retrieve matched features for the input sample
- The average of these missing-modality features in **top-K samples** fills the representation of missing modalities

**Missing Modality Feature Generation** (Wang et al., 2023b): Replaces missing modality features by **averaging the representations of available modalities**, assuming similar feature distributions across modalities.

#### Arithmetic Operation-Based Representation Composition Methods

![Arithmetic Operation-Based Representation Composition](IMG_20260612_151314.jpg)

*Figure 7 (survey): Arithmetic operation-based representation composition methods. Each modality (Mod 1, Mod 2, Mod 3) is passed through its own DNN (DNN-1, DNN-2, DNN-3) to produce per-modality representations (Representation-1, Representation-2, Representation-3). These are then combined through Arithmetic Operations (e.g., Add, Pooling...) into a single fixed-dimension Representation, which is passed through a DNN to the Downstream Task. The representation of any number of modalities can be combined to a fixed dimension.*

**Mechanism:** Can **flexibly fuse any number of modality representations** through arithmetic operations such as pooling without learnable parameters.

**Specific methods:**
- Zhou et al. (2021); Wang et al. (2023b): Fused features through operations like **average/max pooling or addition**, offering low computational complexity and efficiency
- **Merge operations** (Delgado-Escano et al., 2021): Use a **sign-max function**, which selects the highest absolute value from feature vectors, yielding better results by preserving both positive and negative activation values
- **TMFormer** (Zhang et al., 2024): Introduces **token merging based on cosine similarity**
- Zheng et al. (2024): Focus on selecting key vectors or merging tokens to handle missing modalities
- Sarkar et al. (2025): Propose to perform **weighted fusion** of these pooled features to learn a fixed-size multimodal embedding

**Advantages:** Allow the model to flexibly handle features from any number of modalities; enable learning without introducing new learnable parameters.

**Disadvantages:** Not good at capturing the inter-modality relationships through learning, as methods like selecting the largest vector or the feature with the highest score is difficult to fully represent the characteristics of all modalities.

### 4.2.3 Representation Generation Methods

Compared to modality generation methods, representation generation methods can be **integrated seamlessly into existing multimodal frameworks**. Current methods fall into two categories:

**(1) Indirect-to-task representation generation methods:** Treat modality reconstruction as an auxiliary task during training, helping the model intrinsically generate missing modality representations for downstream tasks. The auxiliary task aids in representation generation during training but is **dropped during inference**, hence "indirect-to-task."

**(2) Direct-to-task representation generation methods:** Train a small generative model to directly map available modality representations to the missing modalities.

#### Indirect-to-Task Representation Generation Methods

![Indirect-to-Task Representation Generation](IMG_20260612_151314.jpg)

*Figure 8a (survey): Indirect-to-task representation generation methods. The architecture is supervised by the loss of two tasks: auxiliary reconstruction and downstream tasks. These two tasks sometimes can be trained separately (first training the reconstruction task, discarding the generator, then training the downstream task — as in ActionMAE, Woo et al., 2023). Data flow: [Mod 1 available] + [Mod 2 missing] + [Mod 3 available] → DNN → Representation → (a) DNN → Downstream Task; (b) DNN → [Mod 1 reconstructed] + [Mod 2 reconstructed] + [Mod 3 reconstructed] → Auxiliary Reconstruction Task.*

**Core architecture:** Typically employ the **"encoder-decoder" architecture**.

During training:
- Modality-specific encoders extract features from available modalities
- Reconstruction decoders reconstruct the missing modality
- The downstream module is supervised by downstream task loss for prediction

Those reconstruction decoders are **discarded during inference**, where predictions rely on both the generated missing modality representations and the existing ones.

**Multi-Task Learning approach:** Training both downstream tasks and auxiliary tasks (modality reconstruction) simultaneously.

**Masked Autoencoder-inspired approaches** (He et al., 2022): Some methods split reconstruction and downstream task training.

**Sub-types based on fusion stage:**

**Before Fusion methods** — receive the features from modality-specific encoders directly into each specific reconstruction decoder:
- **MGP-VAE** (Hamghalam et al., 2021): Uses output of the VAE encoder of each modality as the input of each reconstruction decoder and the fused output for Gaussian Process prediction
- **PFNet** (Zheng et al., 2021): Employs RGB and thermal-infrared modalities for pedestrian re-identification, feeding outputs of each encoder into corresponding decoders before fusing them together
- Li et al. (2024b): Adopted a similar approach for brain tumor segmentation, reconstructing each modality before fusing encoder outputs

**After Fusion methods** — use the fused features from modality-specific encoders as the input of reconstruction decoder:
- **Multimodal Factorization Model** (Tsai et al., 2018): Modality-specific encoder-decoders reconstruct the missing modality; the downstream task module employs a separate multimodal encoder-decoder
- Chen et al. (2019); Li et al. (2025a): Decoupled the encoders for reconstruction and downstream tasks, showing that disentangling features helped eliminate irrelevant features
- Jeong et al. (2022): Proposed a simpler approach using multi-level skip connections in a U-Net architecture; achieves better generalization in brain tumor segmentation than Chen et al. (2019)
- Zhou (2023); Sun et al. (2024a): Demonstrated effective cross-modal attention fusion techniques for enhancing reconstruction and segmentation
- **ActionMAE** (Woo et al., 2023); **M3AE** (Liu et al., 2023a): First train reconstruction tasks, then fine-tune the pre-trained encoders for downstream tasks

**Critical limitation:** These methods still **require the missing modality data to be present during training** to supervise the reconstruction decoders.

#### Direct-to-Task Representation Generation Methods

![Direct-to-Task Representation Generation](IMG_20260612_151314.jpg)

*Figure 8b (survey): Direct-to-task representation generation methods. A dedicated modality generation network "GEN-2" directly maps available modality representations to the missing modality representation during both training and inference, without reconstructing the raw modality data. Data flow: [Mod 1 available] → DNN-1 → Representation-1; [Mod 2 missing] → GEN-2 → Representation-2; [Mod 3 available] → DNN-3 → Representation-3. All three representations → DNN → Representation → Downstream Task.*

**Definition:** Use simple generative models to directly map information from available modalities to the missing modality representation. Does not require reconstruction of the missing modality data itself.

**Historical progression:**

- **Hallucination Network (HN)** (Hoffman et al., 2016): Inspired by early work using Gaussian Processes to "hallucinate" missing modalities (Mario Christoudias et al., 2010); predicts missing depth modality features from RGB images; aligns intermediate layer features with another network trained on depth images using L2 loss. Extended to:
  - Pose estimation (Choi et al., 2017)
  - Land cover classification (Kampffmeyer et al., 2018)

- **Translation module** for sentiment analysis (Huan et al., 2023): Maps available-modality representations to the missing ones, with L2 loss applied to align outputs with those from networks trained on missing modalities

- **SMIL** (Ma et al., 2021c): Employs **Bayesian learning and meta-regularization** to directly generate missing modality representations; shows strong generalization on datasets with a high rate of missing modality samples

- Li et al. (2022d): Addressed distribution inconsistency, failure to capture specific modality information, and lack of correspondence between generated and existing modalities

- Cross-modal distribution transformation methods (Wang et al., 2023d): Align distributions of generated and existing modality representations, enhancing discriminative ability

- **RMM — Relation-aware Missing Modal Generator** (Park et al., 2024): For audio-visual question answering (AVQA); consists of a visual generator, auditory generator, and text generator; generates missing auditory representations by analyzing the correlations between visual and textual modalities and utilizing learnable parameter vectors (slots) for reconstruction and synthesis

- **Missing Modality Generation Module** (Guo et al., 2024): For sentiment analysis under missing modalities; maps prompts of available modalities to prompts of missing modalities through a set of projection layers

- **Uncertainty Estimation Module** (Lin et al., 2024): Identifies useless tokens in different modality tokens to facilitate better use of U-Adapter-assisted pre-trained models when downstream tasks face missing modalities

- Liao et al. (2025): Evaluate capabilities of various existing multimodal segmentation models under various conditions of sensor impairment

- **Multi-modal Mixture of Expert** (Park et al., 2025): Enables the model to dynamically handle multi-modality; allows different branches to focus on their own modalities by sharing a query

**Key advantage of direct methods:** Avoid the cumbersome reconstruction of modality data; feature generators are easier to train and integrate into multimodal models.

**Overall limitation of representation generation methods:** Limited by the imbalance of missing modalities in the training dataset or the limited number of full-modality samples, which may lead to training procedures that **overfit to the existing modalities**. Additionally, indirect-to-task methods generally achieve better performance than direct-to-task methods because they can simulate arbitrary missing modality cases during training by dropping modalities.

---

# 5. Methodologies in Strategy Design Aspect

## 5.1 Architecture-Focused Models

Different from above methods of handling missing modality problems at the modality or representation level, many researchers **adjust the model training or testing architecture** to adapt to the missing-modality cases.

Four categories according to their core contributions in dealing with missing modalities:
1. Attention-based methods
2. Distillation-based methods
3. Graph learning-based methods
4. Multimodal large language models

### 5.1.1 Attention-Based Methods

**Background — Self-Attention Mechanism** (Vaswani et al., 2017):
- Each input is linearly transformed to generate **Query, Key, and Value vectors**
- Attention weights are computed by multiplying the query of each element with the keys of others
- Followed by **scaling and softmax** to ensure the weights sum to 1
- A **weighted sum of the values** generates the output

**Two categories of attention-based MLMM methods:**

**(1) Attention Fusion Methods:** Put attention on modality fusion, integrating multimodal information; do not rely on any specific model type; can be suitable for various model types as input and output dimensions are the same.

**(2) Transformer-Based Methods:** Stack attention layers to handle large-scale data with global information capture and parallelization.

#### Attention Fusion Methods

Classified into **intra-modality** and **inter-modality** attention methods:

**Intra-Modality Attention Methods:**

![Intra-Modality Attention Methods](IMG_20260612_151314.jpg)

*Figure 9a (survey): Intra-modality attention methods. Each modality has its own attention module (Intra-Atten-1, Intra-Atten-2, Intra-Atten-3) sharing a Fusion Representation (e.g., learnable query or bottleneck tokens). When modality 2 is missing, its Intra-Atten-2 block is skipped. Since each Intra-Atten of different modalities shares the fusion representation, after calculating them one by one, the output representation is regarded as the fusion of available modalities. Data flow: [Mod 1] → DNN-1 → Representation-1 → Intra-Atten-1; [Mod 3] → DNN-3 → Representation-3 → Intra-Atten-3; Fusion Representation shared → Combined Representation → Downstream Task.*

Compute attention for each modality **independently** before fusing them. This approach focuses on relationships within a single modality; fusion between modalities is achieved by **sharing partial information**.

- **BEV-Evolving Decoder** (Ge et al., 2023): Handles sensor failures in 3D detection by sharing same BEV-query with each modality-specific attention modules, allowing fusion of **any number of modalities**
- Lee et al. (2023a): Proposed modality-aware attention to perform intra-modality attention and predict decisions by using shared **bottleneck fusion tokens** (Nagrani et al., 2021)

**Inter-Modality Attention Methods:**

![Inter-Modality Attention Methods](IMG_20260612_151314.jpg)

*Figure 9b (survey): Inter-modality attention methods. A "Missing Modality Mask" is generated according to the missing modality 2, which helps the inter-modality attention module ignore modality 2. By setting the tokens of modality 2 to zero or negative infinity through masking, the attention mechanism is forced to ignore the missing modality. Data flow: [Mod 1] → DNN-1 → Representation-1; [Mod 3] → DNN-3 → Representation-3; [Mod 2 missing] → Missing Modality Mask → Inter Atten (Ignore Modality 2) → Representation → Downstream Task.*

Often based on **masked attention**: treat missing modality features as masked vectors (using zero or negative infinity values) to better capture dependencies across available modalities.

Unlike conventional cross-modal attention, **masked attention models share the same parameters across all embeddings**, allowing flexible handling of missing modalities.

- Qian & Wang (2023): Designed an **attention mask matrix** to ignore missing modalities, improving model robustness
- **DrFuse** (Yao et al., 2024): Decouples modalities into specific and shared representations, using the shared ones to replace missing modalities, with a custom mask matrix to help the model ignore the specific representations of the missing modality

#### Transformer-Based Methods

Two types based on parameter training scope:

**Joint Representation Learning (JRL):** Full parameter training.

Since transformers have long context lengths to handle many feature tokens, the multimodal transformer can **learn joint representations from any number of modality tokens**.

![Transformer Joint Representation Learning](IMG_20260612_151314.jpg)

*Figure 10 (survey): Joint representation learning methods. Missing modality 2 tokens are replaced by specific masked tokens. Modality encoders (red dashed box) can be replaced by linear projection layers in some methods. Each modality (Mod 1, Mod 2 with masked tokens, Mod 3) goes through its encoder (DNN-1, DNN-2, DNN-2), producing tokens. These are fed into a Multimodal Transformer, which outputs a combined Representation for the Downstream Task.*

Specific methods:
- Gong et al. (2023): Transformer-based fusion module with **flexible number of modality tokens** and a cross-modal contrast alignment loss for egocentric multimodal tasks
- Mordacq et al. (2024): Leveraged **Masked Multimodal Transformers**, treating missing modalities as masked tokens for robust JRL
- Ma et al. (2022): Proposed **optimal fusion strategy search** within the multimodal transformer to find the best fusion method for different missing/full-modality representations
- Radosavovic et al. (2024): Introduced specially designed **mask tokens for an autoregressive transformer** to manage missing modalities in real-world scenarios

**Parameter Efficient Learning (PEL):** Small amount of parameter fine-tuning.

**(a) Prompt tuning:** Optimizes input prompts while keeping model parameters fixed.
- Jang et al. (2024): Introduced **modality-specific prompts** that merge when all modalities are present and allow the model to update all learnable prompts during training
- Liu et al. (2024b): Proposed **Fourier Prompt** — uses fast Fourier transform to encode global spectral information of available modalities into learnable prompt tokens, enabling cross-attention with feature tokens to address missing modalities

**(b) Adapter tuning:** Involves inserting **lightweight adapter layers** (e.g., MLPs) into pre-trained models to adapt to new tasks without modifying the original parameters.
- Qiu et al. (2023): Used a classifier to identify different missing-modality cases and used intermediate features of that classifier as the missing-modality prompt to cooperate with lightweight Adapters. Only needed **1% of the model parameters** to be fine-tuned for downstream tasks
- Reza et al. (2023): Introduce different adapter layers for each missing modality case

**Summary of attention-based method limitations:**
- Although attention-based fusion mechanisms can effectively help deal with the missing modality problem in any framework, none of them care about the missing modality that may **contain important information** required for prediction
- JRL methods are often limited by a **large amount of computing resources** and require relatively large datasets
- PEL methods can achieve efficient fine-tuning, but their performance is still **not comparable** to that of JRL

### 5.1.2 Distillation-Based Methods

**Knowledge distillation** (Hinton et al., 2015): Transfers knowledge from a **teacher model** to a **student model**. The teacher model, with access to more information, helps the student reconstruct missing modalities.

**Two types of distillation methods:**

#### Representation-Based Distillation Methods

![Representation Distillation Methods](IMG_20260612_151314.jpg)

*Figure 11 (survey): Representation distillation methods. Here modality 2 is missing. The teacher DNN has access to all three modalities (Mod 1, Mod 2, Mod 3) and produces Representation-T and output for Downstream Task. The student DNN receives Mod 1 and Mod 3 only and produces Representation-S. Intermediate Distillation (using features from any combination of intermediate layers) and Response Distillation (using logits) both transfer knowledge from teacher to student. The student must learn to approximate full-modality performance despite missing Mod 2.*

Transfer rich representations from the teacher model to help the student capture and reconstruct missing modality features. Classified by whether they use **logits** or **intermediate features**:

**Response distillation methods** — transfer teacher model's logits to students:
- Wang et al. (2020a): Trained modality-specific teachers for missing modalities, then used their **soft labels** to guide a multimodal student
- Hafner & Ban (2023): Employed logits from a teacher model trained with optical data to supervise a reconstruction network for approximating missing optical features from radar data
- **Modality-Aware Distillation** (Saha et al., 2024): Leverages both global and local teacher knowledge in a **federated learning setting**

**Intermediate distillation methods** — align intermediate features between teacher and student:
- Shen & Gao (2019): Used **Domain Adversarial Similarity Loss** to align intermediate layers of the teacher and student, improving segmentation in missing modality settings
- Zhang et al. (2023b): Applied intermediate distillation in endometriosis diagnosis by distilling features from a TVUS-trained teacher to a student using MRI data
- **MultiPro** (Xu et al., 2025): Learns the distribution of the missing modality by distilling unimodal knowledge into learnable unimodal prompts for each modality from all remaining samples available of this modality
- **FedMobile** (Liu et al., 2025a): Tackles the missing-modality problem in multimodal federated mobile sensing systems; a knowledge-contribution-aware federated learning framework that reconstructs missing features through cross-node multimodal knowledge transfer and robust aggregation

#### Process-Based Distillation Methods

Focus on **overall distillation strategies** rather than direct representation transfer.

**Mean Teacher Distillation (MTD)** (Tarvainen & Valpola, 2017):

![Mean Teacher Distillation](IMG_20260612_151314.jpg)

*Figure 12a (survey): Mean Teacher Distillation methods. "EMA" means exponential moving average. The Mean Teacher DNN is a weighted average of the student model's parameters (updated via EMA). Both teacher and student receive the same available modalities (Mod 1, Mod 3). The teacher produces Representation-T; the student produces Representation-S. A Loss is computed on the logits/representations between teacher and student. The teacher provides a stable training signal for the student to learn from.*

Enhances stability by using the **exponential moving average** of the student model's parameters as a teacher.
- Chen et al. (2024b): Applied this to missing modality sentiment analysis, treating missing samples as augmented data
- Li et al. (2022c): Used MTD for Lidar-Radar segmentation, improving robustness against missing modalities

**Self-distillation:**

![Self-Distillation Methods](IMG_20260612_151314.jpg)

*Figure 12b (survey): Self-distillation methods. Models act as both teachers and students, using their own soft labels/representations from each branch to refine themselves during training. Multiple Student/Teacher DNN branches (DNN-1 through DNN-n) each produce Representation-T/S-n. Self-Distillation occurs across branches, leading to the Downstream Task. Each branch learns from the other branches' representations.*

Helps a model improve by **learning from its own soft representations**.
- **ShaSpec** (Wang et al., 2023a): Utilizes distillation between modality-shared branches and modality-specific branches of available modalities
- **Meta-learned Cross-modal Knowledge Distillation** (Wang et al., 2024a): Based on ShaSpec; further weighs the importance of different available modalities
- Ramazanova et al. (2024): Used mutual information and self-distillation for egocentric tasks, making predictions invariant to missing modalities
- **PASSION** (Shi et al., 2024): Self-distillation method designed to use the multi-modal branch to help other uni-modal branches of available modalities to improve multimodal medical segmentation performance with missing modality

#### Hybrid Distillation Methods

Combine **various distillation approaches** to improve student performance:
- Yang et al. (2022): Distilled teacher model knowledge (including logits and intermediate features) at every decoder layer, outperforming ACNet (Wang et al., 2021)
- **ProtoKD** (Wang et al., 2023c): Captures inter-class feature relationships for improved segmentation under missing modality conditions
- **CorrKD** (Li et al., 2024a): Leverages **contrastive distillation and prototype learning** to enhance performance in uncertain missing modality cases

**Critical limitation of distillation methods:** Most assume the complete dataset is available for training (teacher can access full-modality samples during training), with missing modalities encountered only during testing. Therefore, **most distillation methods are unsuitable for handling incomplete training datasets**.

### 5.1.3 Graph Learning-Based Methods

Graph-learning based methods leverage **relationships between nodes and edges in graph-structured data** for representation learning and prediction.

Two main types for addressing the missing modality problem:

#### Graph Fusion Methods

![Graph Fusion Methods](IMG_20260612_151314.jpg)

*Figure 13 (survey): Graph fusion methods. Modality 2 is set as the missing modality (white circle in the graph). Features from modality 1 and 3 can be aggregated via the graph fusion method, and fused features maintain the same dimension. Data flow: [Mod 1] → DNN-1 → Representation-1; [Mod 3] → DNN-3 → Representation-3; Graph Fusion structure combines these → DNN → Downstream Task. The graph structure explicitly models inter-modality relationships.*

Integrate multimodal data using a graph structure, making them **adaptable to various networks**.

- Angelou et al. (2019): Proposed a method mapping each modality to a common space using graph techniques to preserve distances and internal structures
- **HGMF** (Chen & Zhang, 2020): Builds complex relational networks using **hyper edges** that dynamically connect available modalities
- Zhao et al. (2022): Developed Modality-Adaptive Feature Interaction for Brain Tumor Segmentation, adjusting feature interactions across modalities based on **binary existence codes**
- Yang et al. (2023a): Proposed a **graph attention based Fusion Block** to adaptively fuse multimodal features, using attention-based message passing to share information between modalities

#### Graph Neural Network (GNN) Methods

![Graph Neural Network Methods](IMG_20260612_151314.jpg)

*Figure 14a (survey): Individual GNN methods. Each modality first passes through DNN or builds its own graph, then they are uniformly processed using GNNs. Data flow: [Mod 1] builds a graph structure; [Mod 3] → DNN-3 → Representation-3; all are combined into Graph Construction → GNN → Downstream Task. The missing modality 2 has no contribution.*

*Figure 14b (survey): Unified GNN methods. First complete the graph from missing-modality samples via approaches like FeatProp (Malitesta et al., 2024), then process it by GNNs. Data flow: [Mod 1 available] + [Mod 2 missing] + [Mod 3 available] → Graph Construction → Graph Completion (fills in missing modality node) → GNN → Downstream Task.*

Directly encode multimodal information into a graph structure, with GNNs used to **learn and fuse** this information.

**Individual GNN methods** (Figure 14a): Extract features using neural networks or GNNs and fuse them for prediction.
- Early approaches (Wang et al., 2015): Employed **Laplacian graphs** to connect complete and incomplete samples
- **DESAlign** (Wang et al., 2024b): Extracts features using neural networks or GNNs and fuses them for prediction

**Unified GNN methods** (Figure 14b): First complete the graph and then use GNNs for prediction.
- **M3Care** (Zhang et al., 2022): Uses adaptive weights to integrate information from similar patients
- Lian et al. (2023): Proposed the **Graph Completion Network**, which reconstructs missing modalities by mapping features back to the input space
- **FeatProp** (Malitesta et al., 2024): Propagates known multimodal features to infer missing ones (recommendation systems)
- **MUSE** (Wu et al., 2024b): Represents patient-modality relationships as bipartite graphs and learns unified patient representations
- Chen et al. (2023b): Introduced **Entity-Level Modality Alignment** to dynamically assign lower weights to missing/uncertain modalities, reducing the risk of misguidance in knowledge graphs
- Liu et al. (2020): Addresses missing-sensor problem in topology-aware IoT systems via a graph-based feature reconstruction module using **neural message passing**
- Chen et al. (2023a): Proposes a **multi-graph convolutional framework** for human activity recognition explicitly handling asynchronous and incomplete IMU, magnetometer, and physiological signals

**Graph method advantages:** Can leverage the graph structure to better capture relationships both within/between modalities and samples.

**Graph method limitations:**
- **Lower efficiency and scalability**
- **Higher development complexity** compared to other approaches
- Currently unsuitable for large-scale datasets

### 5.1.4 Multimodal Large Language Models (MLLMs)

The impressive transformative power of LLMs (like ChatGPT, Bubeck et al., 2023) can be explained by their impressive **generalization capabilities** across many tasks.

MLLMs are designed to handle diverse user inputs across modalities, including cases with missing modalities, leveraging the flexibility of Transformers. In some current MLLM architectures, **LLMs act as feature processors**, integrating feature tokens from different modality-specific encoders and passing the output to task-/modality-specific decoders. This enables LLMs to:
- Capture rich **inter-modal dependencies**
- Naturally carry the ability to **handle any number of modalities**
- Naturally solve the missing modality problem

**Most MLLMs** employ transformer-based modality encoders, such as CLIP, ImageBind (Girdhar et al., 2023), and LanguageBind (Zhu et al., 2023), which encode multimodal inputs into a **unified representation space**.

**Key MLLM examples:**

- **BLIP-2** (Li et al., 2023b): Bridges the modality gap using a lightweight **Querying Transformer**; optimized for Visual Question Answering, Dialogue, and Captioning
- **LLaVA** (Liu et al., 2024a): Enhances visual-language understanding with GPT-4-generated instruction data
- **AnyGPT** (Zhan et al., 2024): Unifies modalities like text, speech, and images using **discrete representations and multimodal projection adaptors**, enabling seamless multimodal interaction
- **NExT-GPT** (Wu et al., 2023b): Unifies modalities for any-to-any multimodal LLM
- **CoDi** (Tang et al., 2024): Introduces **Composable Multimodal Conditioning**, allowing arbitrary modality generation through weighted feature summation
- **Fuyu** (Bavishi et al., 2023) and **OtterHD** (Li et al., 2023a): Discard modality encoders and use **linear transformation directly**

**MLLM limitations:**
- Inconsistent **multi-modal positional encoding**
- **Training difficulty**
- **High GPU resource requirements**
- No specific MLLM benchmarks on missing modality problems have been proposed

---

## 5.2 Model Combinations

Model combinations aim to **use selected models for downstream tasks**. These methods can be categorized into ensemble, dedicated training, and discrete scheduler methods.

### 5.2.1 Ensemble Methods

Ensemble learning methods allow **flexibility in supporting different numbers of expert models** to combine their predictions.

#### Multimodal Model Ensemble Methods

![Multimodal Model Ensemble](IMG_20260612_151314.jpg)

*Figure 15a (survey): Multimodal model ensemble methods. Contains n three-modal DNNs (Multi-Mod DNN-1 through Multi-Mod DNN-n), each producing Representations 1 through n. The Ensemble module combines all n representations to produce the final output for the Downstream Task. Final predictions are based on the outputs of all n DNNs.*

- Early work (Wagner et al., 2011) in sentiment analysis: Employed ensemble learning to handle missing modalities by averaging predictions from uni-modal models
- **Ensemble-based Missing Modality Reconstruction network** (Zeng et al., 2022): Leverages weighted judgments from multiple full-modality models when generated missing modality features exhibit semantic inconsistencies

#### Unimodal Model Ensemble Methods

![Unimodal Model Ensemble](IMG_20260612_151314.jpg)

*Figure 15b (survey): Unimodal model ensemble methods. Each modality is processed by a dedicated unimodal DNN (Uni-Mod DNN-1, Uni-Mod DNN-2, Uni-Mod DNN-3). Only the available modalities contribute to decision-making. Final predictions are produced by aggregating (Ensemble) the outputs of all accessible unimodal DNNs.*

- Yuan et al. (2012): Early studies in multimodal medical image diagnosis found uniformly weighted methods performed better than weighted averaging and voting approaches
- Chen et al. (2022): Proposed a **probabilistic ensemble method** for multimodal object detection; does not require training and can flexibly handle missing modalities through probabilistic marginalization
- Li et al. (2023c): Proposed Uni-Modal Ensemble with **Missing Modality Adaptation**, training models per modality and performing late-fusion training
- Miech et al. (2018): Calculate **feature-based weights** to fuse modalities, where weights reflect feature importance for final predictions

#### Hybrid Ensemble Methods

- **Dynamic Multimodal Fusion (DynMM)** (Xue & Marculescu, 2023): Uses **gating mechanisms** to select uni/multi-modal models dynamically
- **Proximity-based Modality Ensemble (PME)** (Cha et al., 2024): Uses a cross-attention mechanism with an **attention bias** to integrate box predictions from different modalities; can adaptively combine box features from uni-/multi-model models and reduce noise in multi-modal decoding

### 5.2.2 Dedicated Training Methods

Dedicated training methods **assign different tasks to specialized models**.

![Dedicated Training Methods](IMG_20260612_151314.jpg)

*Figure 16 (survey): Dedicated training methods. A DNN Models Library contains models for each possible modality combination (with three modalities, there are seven possible combinations). When modality 2 is missing, the system Selects the dedicated DNN model that handles modality 1 and modality 3 (purple rectangle). This model then processes the available modalities to produce a Representation for the Downstream Task. Colored circles in purple rectangles represent the modalities each model can handle.*

With three modalities, there are a total of **seven models** with different modality combinations.

- **KDNet** (Hu et al., 2020): First dedicated method proposed to handle different combinations of missing modalities; treats uni-modality specific models as student models, learning multimodal knowledge from the features and logits of a multimodal teacher model
- **ACNet** (Wang et al., 2021): Also utilizes distillation method but introduces **adversarial co-training**, further improving performance
- Lee et al. (2023b): Proposed **missing-modality-aware prompts** for prompt learning using input- and attention-level prompts for each kind of missing modality case; needed only **1% of model parameters** to be fine-tuned
- Reza et al. (2023): Introduce **different adapter layers** for each missing modality case

### 5.2.3 Discrete Scheduler Methods

![Discrete Scheduler Methods](IMG_20260612_151314.jpg)

*Figure 17 (survey): General procedure of discrete scheduler methods. A User provides a Text Command to a Large Language Model (LLM). The LLM "Calls & Schedules" appropriate modules from a Downstream Modules Library (Module 1 through Module n). The system processes the Data Sample (with Mod 1 available, Mod 2 missing, Mod 3 available) through the selected modules and returns results to the User. The LLM acts as an intelligent controller, selecting which modules to invoke based on the available modalities and the task requirements.*

In discrete scheduler methods, **LLMs act as controllers**, determining the execution order of different discrete steps broken down from the major task/instruction.

**Key property:** While the LLM does not directly process multimodal data, it interprets language instructions and orchestrates task execution across uni- and multimodal modules.

As long as there are sufficiently diverse downstream task modules, it is possible to handle the same downstream tasks with **different numbers of modalities**.

**Specific systems:**

- **Visual ChatGPT** (Wu et al., 2023a): Integrates multiple foundation models with ChatGPT to enable interaction through text or images; allows users to pose complex visual questions, provide editing instructions, and receive feedback within a multi-step collaboration framework
- **HuggingGPT** (Shen et al., 2024): LLM-driven agent that manages and coordinates various AI models from Hugging Face; leverages LLMs for task planning, model selection, and summarization to address complex multimodal tasks
- **ViperGPT** (Surís et al., 2023): Combines visual and language tasks into modular subprograms, generating and executing **Python code** for complex visual queries without additional training
- **MM-REACT** (Yang et al., 2023b)
- **LLaVA-Plus** (Liu et al., 2023c)

**Limitations of discrete scheduler methods:**
- Can solve a variety of tasks when there are sufficient types of downstream modules
- Usually require LLMs to **respond quickly** and understand human instructions accurately in real world scenarios

---

# 6. Methodology Discussion & Technical Evolution

## 6.1 Comprehensive Pros and Cons Table

| Aspect | Type | Method | Pros and Cons |
|---|---|---|---|
| **Data Processing** | Modality Imputation | Modality Composition | **Pros:** Simple and effective for data augmentation. **Cons:** Unsuitable for pixel-level tasks (Composition); Modality number ↑, generative model number or training complexity ↑. |
| **Data Processing** | Modality Imputation | Modality Generation | (See above row — combined in survey) |
| **Data Processing** | Representation Focused | Coordinated-Representation | **Pros:** Flexible for novel modalities with better generalization. **Cons:** Hard to balance constraints and handle dataset imbalance. |
| **Data Processing** | Representation Focused | Representation Composition | (See above row — combined in survey) |
| **Data Processing** | Representation Focused | Representation Generation | (See above row — combined in survey) |
| **Strategy Design** | Architecture Focused | Attention-based | **Pros:** Attention methods scale well; Distillation and graph learning are effective. **Cons:** Attention methods often demand large computation & datasets; Distillation and graph learning struggle with incomplete datasets and training complexity. |
| **Strategy Design** | Architecture Focused | Distillation-based | (See above row — combined in survey) |
| **Strategy Design** | Architecture Focused | Graph Learning-based | (See above row — combined in survey) |
| **Strategy Design** | Architecture Focused | Multimodal Large Language Model | (See above row — combined in survey) |
| **Strategy Design** | Model Combinations | Ensemble | **Pros:** Effective for specific tasks. **Cons:** Grows complex with more modalities; Suffers from modality imbalance; Demands significant storage space. |
| **Strategy Design** | Model Combinations | Dedicated | (See above row — combined in survey) |
| **Strategy Design** | Model Combinations | Discrete Scheduler | (See above row — combined in survey) |

## 6.2 Recovery and Non-Recovery Methods

An alternative common taxonomy classifies MLMM methods as **recovery** or **non-recovery**, based on three stages: early, intermediate, and late stage (from the conventional taxonomy).

### Recovery vs. Non-Recovery Stage Analysis

| | Early | Intermediate | Late | Hybrid |
|---|---|---|---|---|
| **Recovery** | Modality Composition (Yang et al., 2024); Modality Generation (Wang et al., 2024c) | Representation Composition (Wang et al., 2023b); Representation Generation (Woo et al., 2023); Distillation-based (Shen & Gao, 2019); Graph Learning-based (Malitesta et al., 2024) | Representation Generation (Ma et al., 2021c); Distillation-based (Li et al., 2022c) | Coordinated-Representation (Ma et al., 2021b); Distillation-based (Sikdar et al., 2023) |
| **Non-Recovery** | Dedicated (Wang et al., 2021); Discrete Scheduler (Wu et al., 2023a) | Representation Composition (Delgado-Escano et al., 2021); Attention-based (Ge et al., 2023); Graph Learning-based (Zhao et al., 2022); Multimodal Large Language Model (Wu et al., 2023b) | Coordinated-Representation (Liang et al., 2019); Distillation-based (Shi et al., 2024); Ensemble (Chen et al., 2022) | Not Applicable |

### Statistical Analysis (from 315 papers):

- **75.5%** of works focus on **recovering** missing modality information
- **24.5%** explore inference without modality recovery
- Among recovery methods:
  - Early-stage: **20.3%**
  - Intermediate-stage: **45.8%** (largest proportion)
  - Late-stage: **4.7%**
  - Multi-stage: **4.7%**
- Among non-recovery methods:
  - Early: **4.2%**
  - Intermediate: **14.1%**
  - Late: **6.3%**

**Why intermediate-stage recovery dominates:** Recovering features (as opposed to the raw data) can avoid much noise and bias while providing more modality-specific/shared information. Additionally, compared to later-stage features, intermediate-stage features tend to be more enriched.

## 6.3 Technical Evolution Discussion (2014–2025)

The development of MLMM methods has closely followed **broader trends in machine learning and deep learning**, with distinct methodological phases:

### Phase 1: Early Heuristic Era (pre-2017)
- Dominated by **data-driven heuristics and ensemble-style solutions**
- Missing modalities addressed by retrieving similar samples from the dataset or by training separate unimodal models and combining them through ensemble strategies
- Corresponds to initial emergence of **Modality Imputation and Model Combinations**

### Phase 2: Representation Learning Era (2017 onward)
- Rise of deep learning frameworks (PyTorch, TensorFlow) and architectures such as ResNet
- Community shifted toward **representation learning** → rapid growth of Representation-Focused Models
- Most stable and persistent line of research since 2017
- Learning aligned, shared, or generative representations enabled models to remain effective even when certain modalities were absent
- Modality-level imputation methods evolved from simple sample-based matching → deep generative models (first autoencoders and GANs, later diffusion models)

### Phase 3: Architecture-Focused Era (post-2018)
- Driven by popularity of **attention mechanisms, deep graph neural networks, and knowledge distillation**
- These techniques allowed models to dynamically adapt to arbitrary combinations of available modalities
- Led to the rise of **Architecture-Focused Models** — category that grows sharply in the statistics
- These methods reframed missing modalities not merely as a data problem but increasingly as an **architectural and training-strategy challenge**
- Enabling flexible fusion, conditional routing, and intra/inter-modal knowledge transfer

### Phase 4: Foundation Model Era (most recent)
- Emergence of **transformer-based foundation models**
- Applying large pretrained models to missing-modality conditions by using them as:
  - Feature processors
  - Task schedulers
  - Backbone architectures capable of consuming variable sets of modalities
- Suggests a future where **missing-modality robustness is a built-in capability of general-purpose models** rather than a specialized add-on

### Future of Model Combinations
- Although historically a niche line of work (due to cost of storing and maintaining multiple models), **recent interest in Agentic AI systems** may renew attention to this class of methods
- As agents increasingly integrate multiple specialized tools, dynamic model selection and combination strategies may become more relevant

### Key Cross-Domain Observation
Applications and architectures are only **weakly linked** in this area, because modern deep learning architectures (e.g., attention) are highly versatile and transferable across tasks and applications:
- GAN-based recovery methods used in both breast-cancer prognosis (Arya & Saha, 2021) and visible–infrared person re-identification (Li et al., 2022d)
- Masked modeling strategy in human action recognition (Woo et al., 2023) closely parallels those used in brain-tumor segmentation (Liu et al., 2023a)

---

# 7. Applications and Datasets

The collection of multimodal datasets is often **labor-intensive and costly**. In certain specific application directions, issues such as user privacy concerns, sensor malfunctions on data collection devices, and other factors can result in datasets with missing modalities. In severe cases, **up to 90% of the samples** may have missing modalities.

## 7.1 Affective Computing

**Goal:** Classify the current emotional state by combining information from multiple modalities such as textual, auditory, and RGB information.

**Applications:** Market research, health monitoring, advertising.

**Historical context:** Early researchers discovered that facial detection algorithms sometimes failed to capture faces (due to occlusion) or the recorded audio was too noisy to use (Cohen et al., 2004; Sebe et al., 2005).

### Dataset Categories in Affective Computing

**Type 1 — Video/Audio/Text/Biosensors for emotional state:**
- **eNTERFACE'05** (Martin et al., 2006): Vision + Audio
- **RAVDESS** (Livingstone & Russo, 2018): Vision + Audio
- **CREMA-D** (Cao et al., 2014): Vision + Audio
- **Ulm-TSST** (Stappen et al., 2021a;b): Vision + Bio-Sensors
- **RECOLA** (Ringeval et al., 2013): Vision + Bio-Sensors
- **SEED** (Zheng & Lu, 2015): Vision + Bio-Sensors
- **CMU-MOSI** (Zadeh et al., 2016): Vision + Audio + Text
- **CMU-MOSEI** (Zadeh et al., 2018): Vision + Audio + Text
- **IEMOCAP** (Busso et al., 2008): Vision + Audio + Text
- **MSP-IMPROV** (Busso et al., 2016): Vision + Audio + Text
- **MELD** (Poria et al., 2018): Vision + Audio + Text
- **CH-SIMI** (Yu et al., 2020): Vision + Audio + Text
- **ICT-MMMO** (Wöllmer et al., 2013): Vision + Audio + Text
- **YouTube** (Morency et al., 2011): Vision + Audio + Text

**Type 2 — Text and images for emotional states (e.g., comics):**
- **eBDtheque** (Guérin et al., 2013): Vision + Text
- **DCM** (Nguyen et al., 2018): Vision + Text
- **Manga 109** (Matsui et al., 2017): Vision + Text

## 7.2 Medical Applications

Medical domain requires comprehensive analysis from:
- Medical history
- Physical examination
- Imaging data
- Cell and gene data

**Focused research areas:**

- **Neuroimaging and Brain Disorders** (Menze et al., 2014; Myronenko, 2019; Young et al., 2017; Jack Jr et al., 2008; Lesjak et al., 2018)
- **Cardiovascular** (Liu et al., 2016; Moody & Mark, 2001)
- **Cancer Detection** (Arya & Saha, 2020; Tomczak et al., 2015; Cheerla & Gevaert, 2019; Farooq et al., 2025)
- **Women Health Analysis** (Zhang et al., 2023b; Xu et al., 2021)
- **Ophthalmology** (Li et al., 2021; Zhang et al., 2022)
- **Sleep Disorders** (Lee et al., 2022; Marcus et al., 2013)
- **Clinical Predictions** (Johnson et al., 2016; 2023; 2019)
- **Biomedical Analysis** (Chaptoukaev et al., 2024; Schmidt et al., 2018b; Kemp et al., 2000)
- **Computational Pathology** (Cui et al., 2022; Peng et al., 2024; Wang et al., 2025a; Xu et al., 2025)
- **Radiology** (Chen et al., 2025; Cui et al., 2022; Liang et al., 2025)
- **Gene and Cell Representation Learning** (Schmauch et al., 2020; Ghahremani et al., 2022; Liu et al., 2025b; Palma et al., 2024; Tu et al., 2022)

### Medical Datasets

| Modality Type | Datasets |
|---|---|
| MRI, CT Scans | BRATS-series (Menze et al., 2014; Myronenko, 2019), ADNI (Jack Jr et al., 2008), IXI |
| Bio-Sensors | StressID (Chaptoukaev et al., 2024), PhysioNet (Kemp et al., 2000), TCGA, BCData (Huang et al., 2020), NuClick (Koohbanani et al., 2020), LYON19 (Swiderska-Chadaj et al., 2019) |
| Electronic Health Record | MIMIC-CXR (Johnson et al., 2019), MIMIC-IV (Johnson et al., 2023), NCH (Lee et al., 2022), BCNB (Xu et al., 2021), ODIR (Li et al., 2021) |

## 7.3 Information Retrieval

**Applications:** Search engines, social media recommendation.

**Challenge drivers:** User privacy concerns and data sparsity issues.

### Information Retrieval Datasets

| Modality Type | Datasets |
|---|---|
| Vision + Text | Amazon Series Datasets (Lakkaraju et al., 2013; McAuley et al., 2015; He & McAuley, 2016; PromptCloud, 2017), MIR-Flickr25K (Duin), NUS-WIDE (Chua et al., 2009) |
| Vision + Audio + Text | MSR-VTT (Xu et al., 2016), TikTok (Lin et al., 2023) |

## 7.4 Remote Sensing

**Applications:** Assessment and analysis of environmental conditions, resources, and disasters on earth from satellites or aircraft. Capabilities for environmental protection, resource management, and disaster response.

**Challenge:** In practice, optical modalities may be unavailable due to **cloud cover or sensor damage**.

### Remote Sensing Datasets

| Modality Type | Datasets |
|---|---|
| Vision | WHU-OPT-SAR (Li et al., 2022b), DFC2020 (Yokoya et al., 2020), MSAW (Shermeyer et al., 2020), Trento (Rasti et al., 2017), Houston (Piacentini, 2023a), Berlin (Hong et al., 2021), GRSS (Debes et al., 2014) |

## 7.5 Pervasive Computing

**Dataset categories by task:**

**(1) Human Activity Recognition:** Focuses on fine-grained activity recognition, gesture understanding, and daily-living behavior modeling. Sensors capture heterogeneous, body-mounted signals (accelerometry, gyroscope, magnetometer, EMG) in naturalistic environments that inherently involve **sensor dropout, misalignment, and variable sampling**.

Datasets:
- **Opportunity** (Roggen et al., 2010): Motion sensors
- **RealDisp** (Banos et al., 2014): Motion sensors
- **DSADS** (Barshan & Altun, 2010): Motion sensors
- **PAMAP2** (Reiss, 2012): Motion sensors
- **Shoaib** (Shoaib et al., 2014): Motion sensors
- **RealWorld-HAR** (Sztyler & Stuckenschmidt, 2016): Motion sensors
- **w-HAR** (Burzer et al., 2025): Motion sensors
- **ERing** (Wilhelm et al., 2015): Motion sensors
- **EMG Physical Action** (Theodoridis, 2011): Motion sensors

**(2) Physiological Monitoring:** Long-term physiological monitoring combining biosignals (ECG, EDA, BVP, RESP, or skin temperature) with motion sensors. Free-living collection protocol results in multimodal recordings with **non-wear intervals, irregular sampling, and missing physiological channels**.

Datasets:
- **WESAD** (Schmidt et al., 2018a): Physiological signals
- **Epilepsy** (Villar et al., 2016): Physiological signals
- **mBrain21** (Donckt, 2024): Physiological signals
- **ETRI Lifelog** (Oh et al., 2025): Physiological signals

**(3) Cognitive State Inference:** Datasets for cognitive-state modeling, mental workload analysis, and motor imagery. EEG's susceptibility to noise, channel corruption, and abrupt signal loss naturally produces **structural missingness at both temporal and spatial (channel) levels**.

Datasets:
- **Self RegulationSCP1** (Birbaumer et al., 1999): EEG
- **EEG-MMID** (Schalk et al., 2004): EEG

## 7.6 Robotic Vision and Sensing

**Modality combination formula:** RGB+X, where X can include LiDAR, radar, infrared sensors, depth sensors, and auditory sensors.

**Five common tasks in robotic vision:**

**(1) Multimodal Segmentation:** Input multimodal data and use segmentation masks to locate objects of interest.
- Common in autonomous vehicle datasets where sensors might fail due to damage or adverse weather conditions
- Typical data: RGB+(Depth/Optical Flow/LiDAR/Radar/Infrared/Event data), RGB+Depth+Thermal, RGB+Angle/Degree of Linear Polarization+Near-Infrared

**(2) Multimodal Detection:** Similar to multimodal segmentation; locates objects using bounding boxes.
- Common: RGB with Depth (Silberman et al., 2012; Lai et al., 2011) and RGB with Thermal (Hwang et al., 2015; kaggle, 2020)

**(3) Multimodal Activity Recognition:** Data from visual sensors plus audio cues.
- Datasets: Ego4D-AR (Grauman et al., 2022), Epic-Sounds (Huh et al., 2023), Epic-Kitchens (Damen et al., 2022), Speaking Faces (John & Kawanishi, 2022), MMG-Ego4D (Gong et al., 2023), Northwestern-UCLA (Wang et al., 2014), NTU60 (Shahroudy et al., 2016)

**(4) Multimodal Person Re-Identification:** Uses depth and thermal information to robustly identify individuals under varying lighting conditions.
- Datasets: RegDB (Ye et al., 2021), SYSU-MM01 (Wu et al., 2017)

**(5) Multimodal Face Anti-Spoofing:** Utilizes multiple sensors (RGB, infrared, depth cameras) to detect and prevent spoofing in facial recognition systems.
- Purpose: Enhance ability to identify fake faces (photos, videos, masks, etc.)
- Datasets: MVSS (Ji et al., 2023), various datasets

### Robotic Vision Datasets

| Modality Type | Datasets |
|---|---|
| Vision | RegDB (Ye et al., 2021), SYSU-MM01 (Wu et al., 2017), UWA3DII (Rahmani et al., 2016), Delivery (Zhang et al., 2023a), NuScenes (Caesar et al., 2020), MVSS (Ji et al., 2023) |
| Vision + Audio | Ego4D-AR (Grauman et al., 2022), Epic-Sounds (Huh et al., 2023), Epic-Kitchens (Damen et al., 2022), Speaking Faces (John & Kawanishi, 2022) |
| Vision + Skeleton/Inertial Data | MMG-Ego4D (Gong et al., 2023), Northwestern-UCLA (Wang et al., 2014), NTU60 (Shahroudy et al., 2016) |

## 7.7 Other Applications

- **Conventional audio-visual classification** (Vielzeuf et al., 2018)
- **Multimodal large language models in visual-dialogue** (Bubeck et al., 2023)
- **Audio-visual question answering** (Park et al., 2024)
- **Captioning** (Liu et al., 2023c)
- **Hand pose estimation** using depth images and heat distributions (Tompson et al., 2014; Choi et al., 2017)
- **Knowledge graph completion** utilizing multimodal data with missing modalities (Liu et al., 2019)
- **Multimodal time series prediction** — stock prediction (Luo et al., 2024) and air quality forecasting (Zheng et al., 2015)
- **Multimodal gesture generation** for natural animations (Liu et al., 2022)
- **Multimodal analysis of single-cell data** (Mimitou et al., 2021) in biology

## 7.8 Dataset Statistics and Observations

From the works reviewed:
- **38%** controlled for varying degrees of missing modality rates and tested accordingly
- **62%** used random modality missing rates for training and testing
- Missing modality rate levels:
  - **Mild (<30%):** 72.6% of works
  - **Moderate (30%-70%):** 82.2% of works
  - **Severe (>70%):** 69.9% of works
  - (A single study may perform validation across multiple levels)
- **36.4%** of works trained on **incomplete training datasets** (cannot access missing modality data during training)
- **63.6%** focused on **testing with missing modalities** (training on complete data)

**Critical gap:** The majority of publicly available datasets mentioned above are **complete in terms of modalities**, and naturally occurring datasets with missing modalities are rare.

---

# 8. Open Issues and Future Research Directions

## 8.1 Accurate Missing Modality/Representation Data Generation

Some researchers (Ma et al., 2021c) have shown that recovering, reconstructing, or generating missing modalities or their intermediate representations can improve model robustness when encountering incomplete multimodal inputs.

**Current limitation:** The generated content often contains **artifacts or hallucinations**, stemming either from the generative models themselves or from imperfections and biases in their training data.

**Future direction:** Better understand **informational complementarity and redundancy** across modalities:
- Not all modalities contribute equally to downstream tasks
- Some provide overlapping signals that can be safely inferred
- Others contain unique information that is difficult to reconstruct without introducing bias

**Research agenda:** Explore principled ways to generate accurate, unbiased, and task-relevant missing modalities — while leveraging modality complementarity and avoiding over-reliance on redundant cues.

## 8.2 Recovery or Non-Recovery Methods?

**The fundamental question:** "To recover or not to recover the missing modality."

**Two approaches:**
1. Recovering the missing modal information so that the multimodal model can continue functioning (Wang et al., 2023a; Ma et al., 2021c)
2. Making predictions using only the data modalities available (Hu et al., 2020; Zhang et al., 2024)

**Identified problems:**
- Yao et al. (2024): Recovered modality information might be **ignored or dominated** by the existing modality information due to imbalanced missing rates — the model still relies on available modalities
- Lin et al. (2024): In situations where crucial modality information is missing, the model might be **unduly influenced by less important existing modality information**, resulting in incorrect outcomes

**Open challenge:** Determining under what circumstances recovery or non-recovery methods are more beneficial, and how to measure whether the recovered modality information is not dominated by other modality information or is unbiased.

## 8.3 Benchmarking and Evaluations for Missing Modality Problems

**Current challenges:**
- Many works are trained/tested under **different missing modality settings**, making comparison difficult
- Recent multimodal models are typically trained on **significantly larger and more diverse datasets** than earlier approaches
- Some recent models can **implicitly infer missing information from learned priors** rather than explicit modality alignment
- When comparing results, it is crucial to account for disparities in **dataset scale, diversity, and accessibility**

**Call to action:** Build a work similar to **MultiBench** (Liang et al., 2021), covering common datasets and settings of missing modality problems, to enable more systematic research.

## 8.4 Method Efficiency

**Current problem:** Many MLMM methods are too computationally heavy:
- Model combination methods (Hu et al., 2020; Xue & Marculescu, 2023) require training an independent model for **each modality combination**
- Methods recovering missing modality information typically involve using a **distinct model for each kind of missing information** or a large unified model

**Requirement:** Many multimodal models need to be deployed on **resource-constrained devices** (space robots, disaster-response robots) that cannot accommodate high-performance GPUs and are prone to sensor damage that is difficult/costly to repair.

**Research direction:** Explore efficient and lightweight solutions for MLMM that can operate effectively under these constraints.

## 8.5 Multimodal Streaming/Temporal Data with Missing Modality

**Current limitation:** Only a small portion of research focuses on handling missing modalities in multimodal streaming or temporal data.

**Challenges in streaming contexts:**
- Long sequences of multimodal data (RGB+X video streams, multimodal time-series signals) are common in real-world settings
- Missing modalities often occur **asynchronously**, creating not only spatial information gaps but also **temporal misalignment** between surviving modalities
- This misalignment makes recovery, fusion, and prediction more challenging

**Research agenda:** MLMM needs to address both missing-modality problems and **temporal alignment issues** in streaming data.

## 8.6 Multimodal Reinforcement Learning with Missing Modality

**Applications:** Robotic grasping, drone control, driving decisions.

**Challenge:** Real-world scenarios frequently have **sensor failures or restricted access** to certain modality data, compromising the robustness of the algorithms.

**Current state:** Only a limited number of papers (Vasco et al., 2021; Lee et al., 2021; Awan et al., 2022) aim to address the problem of missing data or modalities in reinforcement learning.

## 8.7 Multimodal AI with Missing Modality for Natural Science

**Application domains:**
- **Drug prediction** (Deng et al., 2020): Integrating diverse data types like molecular structures, genomic sequences, and spectral images
- **Materials science** (Muroga et al., 2023): Predicting diverse properties of advanced materials

**Challenges:**
- Data access restrictions
- High acquisition costs
- Incompatibility of heterogeneous data
- These issues often result in missing modalities

**Current state:** Limited research (He et al., 2024) on MLMM in scientific applications.

## 8.8 Handling Practical Missing Modality Problems in Real-World

**Target environments:** IoT infrastructures, wearables, smart homes, mobile sensing, distributed environmental sensor networks.

**Why these environments are critical:** Missingness is not occasional or synthetic but **structural and persistent** due to:
- Heterogeneous devices
- Limited battery life
- Bandwidth restrictions
- Hardware degradation
- Asynchronous sampling
- Intermittent connectivity

**Multimodal streams exhibit:**
- Long gaps
- Irregular intervals
- Complete sensor dropout

**Empirical findings from mobile and wearable sensing:**
- Human activity recognition systems frequently encounter **inconsistent sampling, device failures, and user-dependent variability** (Nweke et al., 2018)
- Wrist-worn sensing studies report **prolonged non-wear periods** and dropout of inertial or physiological channels (Van Der Donckt et al., 2024)
- Systems integrating IMUs, magnetometers, and physiological sensors exhibit **structural missingness induced by asynchronous fusion** (Chen et al., 2023a)

**Research gaps:**
- MLMM lacks general frameworks capable of handling long-term partial observability, heterogeneous devices, asynchronous and irregular multimodal streams, and multi-agent inconsistencies in a unified manner
- More systematic evaluations and standardized benchmarks are urgently needed

**Complementary future directions:**
- **Resource-aware MLMM robustness strategies**: Lightweight methods such as signal augmentation and cross-modality distillation (Jeon et al., 2021), energy-efficient imputation pipelines for mobile health (Hussein et al., 2024b)
- **Scalable generative approaches**: SensorGAN (Hussein & Bhat, 2024) shows promise for reconstructing missing sensor channels, but broader assessments of scalability, latency, and reliability are lacking

---

# 9. System Architecture & Visual Pipeline

![Model Architecture Diagram](IMG_20260612_151314.jpg)

*Primary model architecture diagram for MLMM systems. This diagram represents the generalized gated fusion pipeline discussed in the survey's taxonomy, illustrating how data streams from multiple sensor modalities flow through individual encoder branches, are aligned in a shared representation space, and are then fused through a gated or attention-based mechanism to produce a final multimodal embedding for downstream task prediction. The architecture is designed specifically to handle missing modality inputs without token suppression — i.e., without simply zeroing out or masking away the information contributed by available modalities.*

## 9.1 Low-Level Technical Architecture: Full Data Flow Pipeline

The following describes the comprehensive data flow through a generalized MLMM system following the architectural principles detailed throughout this survey:

### Stage 1: Modality-Specific Encoding

Each available modality **M_i ∈ {M_1, M_2, ..., M_N}** is passed through a dedicated modality-specific encoder **E_i(·)**:

```
R_i = E_i(M_i)    for all i where M_i is available
```

Where:
- **E_i** may be a convolutional network (for image/depth), transformer encoder (for text), or recurrent network (for time-series)
- **R_i** is the modality-specific representation in a shared latent dimensionality d
- For **missing modalities**, no encoder is invoked — the slot is left unoccupied (no forced zero-fill)

### Stage 2: Missing Modality Detection & Mask Generation

A **missing modality detection module** identifies which modalities are absent:
```
mask_i = 1  if M_i is available
mask_i = 0  if M_i is missing
```

This binary mask is used to gate contributions to the fusion layer. Critically, the mask does **not** suppress or zero-fill available modality tokens — it only signals to the fusion mechanism which slots are empty.

### Stage 3: Representation Alignment Layer

Available representations R_i are projected into a **common semantic alignment space** using either:
- Coordinated constraints (CCA, TRR, HSIC — see Section 4.2.1)
- Learned projection matrices: `R_i_aligned = W_i · R_i + b_i`

The alignment layer ensures that representations from different modalities are **semantically consistent** and comparable, enabling meaningful fusion even when some modalities are absent.

### Stage 4: Gated Fusion Block (Without Token Suppression)

The gated fusion block combines all available aligned representations:

**Attention-based variant (Inter-Modality Attention):**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```
Where Q, K, V are derived from the available modality representations only. Absent modalities contribute neither queries, keys, nor values — they are **not replaced with zero vectors** that would dilute the attention distribution.

**Arithmetic fusion variant:**
```
R_fused = Σ_i (mask_i · w_i · R_i_aligned) / Σ_i mask_i
```
Where w_i are learnable weights (or uniform weights in non-parametric approaches). The denominator normalizes over the **count of available modalities only**, avoiding the bias introduced by including zero vectors in the average.

**Representation generation variant:**
```
R_missing_j = G_j({R_i_aligned : mask_i = 1})
```
A generator G_j maps available modality representations to the representation of missing modality j.

### Stage 5: Downstream Task Prediction

The fused representation **R_fused** is passed to a task-specific head:
```
ŷ = f_task(R_fused)
```
Where f_task may be a classifier, segmentation head, regression module, or generation decoder depending on the application.

## 9.2 Key Design Principles for the Gated Fusion Block

Following the analysis of the survey's twelve method categories, the following principles govern robust gated fusion design:

**Principle 1 — No Forced Zero-Fill of Missing Modalities:**
Replacing absent modalities with zero vectors artificially inflates the batch size while injecting zero-information tokens into the fusion. The preferred approach is to skip absent modalities entirely or to explicitly generate their representations before fusion.

**Principle 2 — Dynamic Input Dimensionality:**
The fusion mechanism must handle any subset of {1, ..., N} modalities. This requires either (a) pooling-based operations that reduce variable-length inputs to fixed-dimension outputs, or (b) attention mechanisms that naturally process variable-length token sequences.

**Principle 3 — Modality Importance Weighting:**
Different modalities contribute unequally to different downstream tasks. Gating weights w_i should either be learned (per-task) or dynamically inferred from the input (attention-derived) to reflect the informational contribution of each available modality.

**Principle 4 — Representation-Level vs. Data-Level Recovery:**
Recovery at the representation level (Section 4.2) is generally preferred over recovery at the data level (Section 4.1) because features carry more generalized information than raw modality data and can avoid much noise and bias.

**Principle 5 — Training vs. Inference Missing-Modality Handling:**
Methods must distinguish between:
- **Training-time missing modalities:** Require data augmentation strategies (dropping modalities), indirect-to-task reconstruction, or distillation from full-modality teachers
- **Inference-time missing modalities:** Require runtime adaptation (masks, generators, attention routing) that does not depend on the missing modality's presence

---

# 10. Conclusion

This survey presents the **first comprehensive survey of Deep Multimodal Learning with Missing Modality**. Key takeaways:

**On methodology:** A total of 354 papers spanning 2012–2025 reveal that the field has evolved from simple data-level heuristics (zero-fill, retrieval) through representation learning (coordinated representations, generation), to sophisticated architecture-level solutions (attention, distillation, graph learning, MLLMs). The intermediate-stage recovery of missing modality features accounts for the **largest proportion of research** (45.8% of recovery methods), because features at this level avoid raw data noise while providing rich modality-specific information.

**On statistics:** 75.5% of papers focus on recovery-based approaches; 63.6% train on complete data and test with missing modalities; only 36.4% address the harder problem of incomplete training datasets — a critical gap the field must address.

**On open challenges:**
1. **Accurate generation** of unbiased, task-relevant missing modality information
2. **When to recover vs. not recover** — a principled decision framework is still lacking
3. **Unified benchmarks** covering diverse datasets, missing rates, and missing patterns
4. **Efficiency** for resource-constrained deployment environments
5. **Streaming/temporal data** with asynchronous modal dropout
6. **Reinforcement learning** with missing sensory inputs
7. **Natural science applications** with inherently incomplete multimodal data
8. **Real-world IoT/wearable** environments with persistent, structural missingness

**The trajectory of the field** points toward a future where missing-modality robustness is not a specialized add-on but a **built-in capability of general-purpose multimodal foundation models** — particularly as transformer-based and MLLM architectures become increasingly capable of processing variable sets of modalities through flexible token-level fusion and long-context processing.

---

## Appendix: Conference and Journal Abbreviations

| Abbreviation | Full Name |
|---|---|
| AAAI | Association for the Advancement of Artificial Intelligence |
| IJCAI | International Joint Conference on Artificial Intelligence |
| NeurIPS | Conference on Neural Information Processing Systems |
| ICLR | International Conference on Learning Representations |
| ICML | International Conference on Machine Learning |
| CVPR | IEEE Conference on Computer Vision and Pattern Recognition |
| ICCV | International Conference on Computer Vision |
| ECCV | European Conference on Computer Vision |
| ACL | Annual Meeting of the Association for Computational Linguistics |
| EMNLP | Conference on Empirical Methods in Natural Language Processing |
| KDD | ACM SIGKDD Conference on Knowledge Discovery and Data Mining |
| ACM MM | ACM International Conference on Multimedia |
| MICCAI | International Conference on Medical Image Computing and Computer-Assisted Intervention |
| ICASSP | IEEE International Conference on Acoustics, Speech and Signal Processing |
| TPAMI | IEEE Transactions on Pattern Analysis and Machine Intelligence |
| TIP | IEEE Transactions on Image Processing |
| TMI | IEEE Transactions on Medical Imaging |
| TMM | IEEE Transactions on Multimedia |
| JMLR | Journal of Machine Learning Research |
| TMLR | Transactions on Machine Learning Research |

---

*Reference manual compiled from: Wu, R., Wang, H., Chen, H.-T., & Carneiro, G. (2026). Deep Multimodal Learning with Missing Modality: A Survey. Transactions on Machine Learning Research. arXiv:2409.07825v4.*
