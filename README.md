# Automated Halo CME Identification Using SWIS Level-2 Data and Machine Learning

A data-driven machine learning study for detecting **Coronal Mass
Ejections (CMEs)** and distinguishing **Halo CMEs** using in-situ
solar-wind measurements from the **Solar Wind Ion Spectrometer (SWIS)**
aboard **Aditya-L1**.

The project combines **SWIS Level-2 plasma measurements** with CME
timestamps and Halo classifications from the **CACTUS CME catalog** to
build automated CME-window identification and Halo-CME classification
pipelines. The study covers **May 2024 -- April 2025**.


------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Scientific Motivation](#scientific-motivation)
-   [Research Objectives](#research-objectives)
-   [Dataset](#dataset)
-   [Data Processing Pipeline](#data-processing-pipeline)
-   [Feature Engineering](#feature-engineering)
-   [Machine Learning Models](#machine-learning-models)
-   [Class Imbalance](#class-imbalance)
-   [Results](#results)
-   [Model Selection](#model-selection)
-   [Limitations](#limitations)
-   [Future Directions](#future-directions)
-   [Research Significance](#research-significance)

------------------------------------------------------------------------

## Overview

Coronal Mass Ejections are large eruptions of plasma and magnetic fields
from the solar corona. They can travel at speeds exceeding **3000 km/s**
and can affect satellites, spacecraft, communication systems, and
terrestrial power infrastructure.

**Halo CMEs** appear as expanding halos in coronagraph observations and
are particularly important because they are associated with
Earth-directed eruptions.

This project investigates whether CME and Halo-CME signatures can be
identified directly from **in-situ solar-wind plasma measurements**,
rather than relying only on remote coronagraph observations. The study
specifically develops machine-learning pipelines around **Aditya-L1 SWIS
data**, addressing a research gap identified in the project.

------------------------------------------------------------------------

## Scientific Motivation

### Why detect CMEs?

CMEs can produce strong disturbances in the solar wind and are important
drivers of space-weather events. Automated identification can contribute
to improved monitoring and potentially support future space-weather
prediction and early-warning systems.

### Why Halo CMEs?

Halo CMEs are particularly significant because their coronagraph
appearance is associated with Earth-directed eruptions. Correctly
distinguishing Halo CMEs from regular CME events is therefore an
important secondary classification problem.

### Research Gap

The project focuses on a relatively underexplored combination of:

-   In-situ solar-wind measurements
-   Aditya-L1 SWIS Level-2 data
-   CACTUS CME event labels
-   Physics-informed feature engineering
-   Machine-learning-based CME and Halo-CME classification

The project identifies the limited use of in-situ plasma measurements
for data-driven CME identification, and particularly the lack of
published ML-based Halo-CME classification using Aditya-L1 SWIS data, as
the motivation for this work.

------------------------------------------------------------------------

## Research Objectives

### 1. CME Window Identification

Automatically identify temporal windows in SWIS time-series measurements
corresponding to CME passages.

CACTUS CME timestamps are used as the reference labels, with temporal
margins introduced to account for propagation and timing uncertainties.


### 2. Halo CME Classification

Distinguish **Halo CME events** from regular CME events using their
in-situ plasma signatures.

This classification is intended to support further research toward
early-warning systems for Earth-directed solar ejections.

### 3. Hybrid Physics-ML Framework

The overall approach combines:

-   Domain knowledge from plasma physics
-   Physics-motivated feature engineering
-   Statistical thresholds
-   Machine-learning pattern recognition
-   Models designed to handle severe class imbalance


------------------------------------------------------------------------

## Dataset

### SWIS Level-2 Data

The primary measurements come from the **Solar Wind Ion Spectrometer
(SWIS)**, part of the **ASPEX (Aditya Solar wind Particle EXperiment)**
instrument aboard the Aditya-L1 spacecraft at the Sun-Earth L1 Lagrange
point. 

### Key Plasma Parameters

The study uses SWIS measurements including:
- Particle flux density: Solar-wind particle flux measurement.
- Number density: Proton number density in particles/cm³.
- Plasma temperature: Plasma temperature in K.
- Solar-wind bulk velocity: Bulk solar-wind velocity in km/s.


### Temporal Coverage

The dataset covers approximately **12 months**, from **May 2024 to 3
April 2025**, with varying observational cadence.


### Ground Truth

CME event timestamps and Halo classifications are obtained from the
**CACTUS automated CME detection catalog**.


------------------------------------------------------------------------

## Data Processing Pipeline

The processing workflow consists of four main stages.

``` text
CACTUS CME Events
       │
       ▼
┌───────────────────────┐
│ 1. Window Extraction  │
│       ± 6 hours       │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 2. Data Cleaning      │
│ 5-minute cadence      │
│ Missing-value handling│
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 3. Feature Generation │
│ Statistics + gradients│
│ Combined metrics      │
└──────────┬────────────┘
           ▼
┌──────────────────────────┐
│ 4. Nonlinear PCA         │
│ Autoencoder latent space │
└────────────┬─────────────┘
             ▼
       ML Classification
```

#### Step 1 --- Window Extraction
For each CACTUS CME event, SWIS measurements are extracted using a
**±6-hour temporal window**. This is intended to capture both pre-event
conditions and post-event signatures. 

#### Step 2 --- Data Cleaning
The measurements are:
-   Resampled to a uniform **5-minute cadence**
-   Interpolated for missing values using physics-consistent methods
-   Processed to remove instrumental artifacts


#### Step 3 --- Feature Generation
Features include:

-   Rolling means
-   Rolling standard deviations
-   Temporal derivatives
-   Flux × speed
-   Density × temperature ratios
-   Other combined physics-motivated metrics


#### Step 4 --- Dimensionality Reduction
An **autoencoder-based nonlinear PCA** is applied to obtain compact
latent representations intended to preserve important plasma dynamics.

## Feature Engineering

The feature engineering is motivated by expected CME-driven plasma
disturbances.

Examples include:

-   **Velocity gradients** → potential signatures of shock fronts
-   **Density spikes** → compressed plasma ahead of CME ejecta
-   **Temperature variations** → transitions between ambient solar wind
    and CME material
-   **Moving averages** → smoothing of instrumental noise while
    preserving larger-scale structures
-   **Combined metrics** → relationships between multiple plasma
    parameters that may characterize coherent CME structures


The goal is therefore not simply to maximize predictive performance, but
to incorporate physically meaningful information into the ML feature
space.

## Machine Learning Models

The project evaluates a broad range of model families.

### LSTM Networks

Long Short-Term Memory networks are used for sequential pattern
recognition in the SWIS time series. Their purpose is to capture
temporal dependencies in the evolution of plasma parameters during CME
passage. 

### AutoML

An automated machine-learning framework explores model architectures,
hyperparameters, and preprocessing strategies and provides a performance
baseline. 

### Gradient Boosting

The study evaluates:

-   XGBoost
-   LightGBM
-   CatBoost

These tree-based ensemble methods are particularly suited to
heterogeneous tabular features and complex feature interactions.

### Shapelet Transform

Shapelet-based methods search for interpretable time-series subsequences
that discriminate between classes. These can provide physically
interpretable temporal patterns associated with CME signatures.

### Statistical Baselines

The project also considers:

-   **Gaussian Discriminant Analysis (GDA)**
-   **Rule-based, physics-guided threshold classifiers**


### Ensemble Methods

Advanced ensemble approaches include:
-   Stacking models
-   Weighted voting

These combine predictions from multiple base models.

------------------------------------------------------------------------

## Class Imbalance

One of the major challenges is the scarcity of Halo CME events.

The reported dataset distribution is approximately just 1% HALO CME events and 95% Non-CME observations.

Full-halo **Type IV** events are especially rare.

This imbalance makes raw accuracy insufficient as the sole evaluation
criterion. The study therefore emphasizes balanced performance and uses
oversampling strategies.

------------------------------------------------------------------------

## Results

### Best Performing Model

The study identifies **CatBoost + SMOTE oversampling** as the preferred
model configuration.

Reported performance:

-   **Accuracy:** 92%
-   **Weighted F1-score:** 0.89

The model was selected for its balanced performance across the
evaluation criteria and its robustness for the heterogeneous feature
space. 

### AutoML

AutoML achieved a higher raw accuracy:

-   **Accuracy:** 99%
-   **F1-score:** 0.84

The lower F1-score suggests possible bias toward the majority class
and/or overfitting, making raw accuracy alone an unreliable measure of
performance for this imbalanced dataset. 

### Other Models

The study reports:

-   **LSTM and other gradient boosting variants:** approximately 85--88%
    accuracy
-   **Shapelet-based methods:** slightly lower predictive performance
    but greater interpretability


------------------------------------------------------------------------

## Model Selection

The final model selection is based on more than raw accuracy.

The reported criteria are:

1.  Balanced precision-recall trade-off
2.  Generalization to unseen events
3.  Computational efficiency
4.  Interpretability of feature importance
5.  Robustness to noise and missing data


This leads to the selection of **CatBoost with SMOTE** as the preferred
architecture.

------------------------------------------------------------------------

## Limitations

### Limited Observation Period

The study uses only a 12-month observation window, which substantially
limits the number of Halo CME examples available for training and
validation. 

### Instrumental Data Gaps

Temporal coverage gaps occur because of instrumental downtime.

### CACTUS--SWIS Timing Uncertainty

Potential uncertainty exists between:

-   The CME detection time in coronagraph observations
-   The actual in-situ arrival time at the spacecraft

CACTUS timestamp precision and systematic biases can therefore affect
the labeling and alignment process. 

### Feature Interpretability

The physical interpretation of engineered features requires deeper
validation. Some threshold choices are currently empirical rather than
derived directly from theory. 

### Limited Halo CME Statistics

The small number of Halo CME events makes synthetic augmentation and
careful validation important for obtaining reliable estimates of model
performance. 

------------------------------------------------------------------------

## Future Directions

The study identifies several directions for strengthening the physics-ML
framework:

-   Validate engineered features using plasma-physics first principles
-   Establish a stronger theoretical basis for feature--CME
    relationships
-   Incorporate magnetic-field measurements when available
-   Develop physics-guided loss functions
-   Increase the observation period to obtain more rare Halo CME events
-   Improve validation strategies for the severely imbalanced dataset


------------------------------------------------------------------------

## Research Significance

This project demonstrates a framework for combining **spacecraft in-situ
measurements, automated CME catalogs, physics-informed feature
engineering, dimensionality reduction, and machine learning** for
automated CME analysis.

The central contribution is the use of **Aditya-L1 SWIS Level-2
measurements** for a data-driven investigation of CME and Halo-CME
signatures. The resulting pipeline provides a foundation for further
research into automated solar-wind event detection and potentially
future space-weather monitoring systems. 

------------------------------------------------------------------------

## Project Scope at a Glance

``` text
                ADITYA-L1 / SWIS
                       │
                       ▼
             SWIS Level-2 Plasma Data
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Data Cleaning              CACTUS CME Catalog
          │                         │
          └────────────┬────────────┘
                       ▼
               Temporal Alignment
                       │
                       ▼
             Feature Engineering
                       │
                       ▼
             Autoencoder NLPCA
                       │
                       ▼
             ML Model Evaluation
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
      LSTM          Boosting         Shapelets
       │          XGB/LGBM/Cat         │
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                Imbalance Handling
                    (SMOTE)
                       │
                       ▼
             CatBoost + SMOTE
                       │
                       ▼
              CME / Halo CME
                Classification
```
