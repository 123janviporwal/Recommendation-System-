# Presentation Advisor Recommendation System

A full replication and improvement study of a research paper implementing five recommendation system architectures for presentation quality advising. All five models **beat the paper's reported baselines**. Built on the [Presentation Advisor Dataset](https://www.kaggle.com/datasets/janvijain96/presentation-advisor-data) and run on Kaggle with GPU (Tesla T4).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model 1 — CBF + CF Hybrid Baseline](#model-1--cbf--cf-hybrid-baseline)
- [Model 2 — CF Autoencoder](#model-2--cf-autoencoder)
- [Model 3 — Dueling DQN + Double DQN + PER](#model-3--dueling-dqn--double-dqn--per)
- [Model 4 — Hybrid Multi-Tower Neural Network](#model-4--hybrid-multi-tower-neural-network)
- [Model 5 — Hybrid + Custom Pre-Trained Embeddings](#model-5--hybrid--custom-pre-trained-embeddings)
- [Results Summary](#results-summary)
- [All Models Comparison](#all-models-comparison)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Project Overview

This project replicates and improves five recommendation models from a research paper on presentation quality recommendation. Each model is built in an independent Kaggle notebook, evaluated against the paper's reported baselines, and improved through architecture changes, hyperparameter tuning, and regularisation techniques.

**Key achievements:**
- All 5 models outperform their respective paper baselines
- Model 4 achieves MAE = **0.0803** vs paper's 0.11 (best performer)
- Model 3 reduces paper DQN MAE by **58%** (3.22 → 1.3448)
- Full EDA, hyperparameter sweeps, and detailed result visualisations for every model

---

## Dataset

**Source:** [Presentation Advisor Data — Kaggle](https://www.kaggle.com/datasets/janvijain96/presentation-advisor-data)

| File | Shape | Description |
|---|---|---|
| `interaction_data.csv` | (25000, 75) | User–article interactions with ratings and all feature columns |
| `ratings_matrix.csv` | (3000, 5) | Pivot-style rating records |
| `presentations.csv` | (78, 15) | Presentation metadata |
| `presentations_df.csv` | — | Processed presentation features |
| `recommendations.csv` | (100, 26) | Article-level metadata: presentation type, audience type, main issue, popularity |
| `user_profiles.csv` | (30, 5) | User type, location, preferences, preferred presentation type |

**Key statistics from EDA:**
- Total ratings: **25,000** — Unique users: **50** — Unique articles: **500**
- Rating range: **2.0 → 5.0**
- Ratings matrix sparsity: **37.10%** (1,887 non-zero out of 3,000 cells)
- User types: business, teacher, student, researcher, manager, technical, specialist
- Presentation types: formal, creative, business, educational, technical, persuasive
- Audience types: academic, business, technical, kids, general
- 18 presentation quality dimensions (Graphics, Readability, Bullets, Infographics, etc.)

---

## Repository Structure

```
.
├── recommendation-paper-model-1.ipynb       # CBF + CF Hybrid Baseline
├── recommendation-paper-model-2.ipynb       # CF Autoencoder
├── recommendationpaper-model-3.ipynb        # Dueling DQN + Double DQN + PER
├── recommendation-paper-model-4.ipynb       # Hybrid Multi-Tower Neural Network
├── recommendation-paper-model-5.ipynb       # Hybrid + Custom Pre-Trained Embeddings
├── Plots/                                   # All saved output plots (see below)
└── README.md
```

### Plots directory — complete list of generated graphs

```
Plots/
│
├── EDA (shared across models)
│   ├── eda_ratings_overview.png             # Rating distributions, ratings per user/article
│   ├── eda_user_preferences.png             # Avg preference scores + correlation heatmap
│   ├── eda_user_presentation_types.png      # User types, presentation types, audience types
│   ├── eda_temporal.png                     # Monthly volume, avg rating over time
│   ├── eda_silhouette.png                   # K-Means silhouette score for user/item clusters
│   └── feature_importance.png              # Pearson correlation of features with rating
│
├── Model 1
│   ├── alpha_sweep.png                      # Validation MAE across alpha values 0.0–1.0
│   ├── k_sweep.png                          # Validation MAE across k values 5–50
│   ├── cbf_cf_results_improved.png          # Predicted vs true, residuals, comparison bar
│   ├── per_rating_error.png                 # MAE breakdown by each true rating value
│   └── summary_dashboard.png               # Metrics comparison + % improvement pie chart
│
├── Model 2
│   ├── tuning_results.png                   # Val MAE and val loss per hyperparameter config
│   ├── autoencoder_results_improved.png     # Loss curve, MAE curve, LR schedule, scatter
│   ├── latent_space_pca.png                 # PCA 2D projection of 64-dim latent space
│   ├── per_user_error_analysis.png          # MAE per user, MAE vs rating count scatter
│   ├── reconstruction_heatmap.png           # Original vs reconstructed rating heatmaps
│   └── all_models_comparison.png           # All 5 models MAE/RMSE/MSE bar comparison
│
├── Model 3
│   ├── dqn_training_dynamics.png            # Episode rewards, Huber loss, val MAE, epsilon
│   ├── dqn_prediction_quality.png           # Predicted vs true, residuals, Q-value spread
│   ├── dqn_vs_paper.png                     # Improved DQN vs paper DQN bar comparison
│   ├── dqn_per_rating_error.png             # MAE per true rating value
│   └── dqn_summary_dashboard.png           # Metrics comparison + hyperparameter table
│
├── Model 4
│   └── hybrid_results.png                   # Loss/MAE curves, tuning, scatter, error dist
│
└── Model 5
    └── hybrid_emb_results.png               # Loss/MAE curves, tuning, scatter, comparison
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn tqdm
```

### 3. Download the dataset

Download the dataset from Kaggle and place all 6 CSV files at:

```
/kaggle/input/datasets/<username>/presentation-advisor-data/
```

> **Note:** Models 1, 2, 3 use username `bh7gy0189` in their paths. Models 4 and 5 use `janvijain96`. Update these paths if running locally.

### 4. Run notebooks

Each notebook is self-contained. Run them independently on Kaggle (GPU recommended for Models 3, 4, 5).

---

## Exploratory Data Analysis

EDA is performed in every notebook. The following plots are generated and saved to `Plots/`.

---

### Ratings & Interactions Overview

<!-- ADD PLOT HERE -->
<!-- File: Plots/eda_ratings_overview.png -->
<!-- Generated by: all notebooks (Model 3 version saved to saved_models/eda_ratings_overview.png) -->
![EDA Ratings Overview](Plots/eda_ratings_overview.png)

**6-panel figure covering:**
- Rating distribution in `interaction_data` (histogram)
- Rating distribution in `ratings_matrix` (histogram)
- Ratings per user (boxplot)
- Ratings per article (histogram / CDF)
- Top 20 articles by average rating
- Rating statistics summary table

---

### User Preferences Analysis

<!-- ADD PLOT HERE -->
<!-- File: Plots/eda_user_preferences.png -->
![EDA User Preferences](Plots/eda_user_preferences.png)

**2-panel figure:**
- Horizontal bar chart of average user preference scores across all 18 quality dimensions
- Full correlation heatmap of user preference features

---

### User Types, Presentation Types & Audience Types

<!-- ADD PLOT HERE -->
<!-- File: Plots/eda_user_presentation_types.png -->
![EDA User and Presentation Types](Plots/eda_user_presentation_types.png)

**6-panel figure:**
- User type distribution (business, teacher, student, researcher, manager, technical, specialist)
- User preferred presentation type
- Article presentation type distribution
- Audience type distribution
- Average rating by preferred presentation type
- Average rating by audience type

---

### Temporal Analysis

<!-- ADD PLOT HERE -->
<!-- File: Plots/eda_temporal.png -->
![EDA Temporal Analysis](Plots/eda_temporal.png)

**3-panel figure:**
- Monthly rating volume (area chart)
- Average rating over time ± 1 standard deviation
- Rolling 3-month correlation between volume and average rating

---

### Silhouette Score & Feature Importance

<!-- ADD PLOT HERE -->
<!-- File: Plots/eda_silhouette.png -->
![Silhouette Score](Plots/eda_silhouette.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/feature_importance.png -->
![Feature Importance](Plots/feature_importance.png)

**Silhouette scores (K-Means on user/item feature matrices):**
- User cluster silhouette: **0.0931**
- Item cluster silhouette: **0.0863**
- Conclusion: user clusters are slightly tighter, justifying a larger user tower in Model 4

**Top 10 features by Pearson correlation with rating:**

| Feature | Correlation |
|---|---|
| user_avg_rating | 0.8216 |
| user_preference_Tables | 0.3657 |
| user_type_researcher | 0.2843 |
| user_preference_Agenda | 0.2796 |
| item_avg_rating | 0.2500 |
| item_rating_std | 0.2282 |
| user_preference_Text size | 0.2158 |
| user_preference_Readability | 0.2059 |
| user_preference_General tips | 0.1896 |
| user_id | 0.1863 |

---

## Feature Engineering

Applied consistently across all notebooks:

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclic encoding of hour of day (0–23) |
| `day_sin`, `day_cos` | Cyclic encoding of day of week (0–6) |
| `month_sin`, `month_cos` | Cyclic encoding of month (1–12) |
| `hours_since_first` / `days_since_first` | Time elapsed since user's first interaction |
| `decay_weight` | Exponential decay: `exp(−λ × days_since_first)`, λ=0.01 |
| `weighted_<problem_col>` | Each problem dimension × decay weight |
| `user_avg_rating` | Per-user mean rating |
| `user_rating_count` | Number of ratings per user |
| `user_rating_std` | Std dev of ratings per user |
| `item_avg_rating` | Per-article mean rating |
| `item_rating_count` | Number of ratings per article |
| `item_rating_std` | Std dev of ratings per article |
| `preference_match_score` | Count of aligned user preference vs article issue dimensions (Model 3 only) |

All continuous features scaled to [0, 1] with `MinMaxScaler` fitted on training set only.

**Final engineered dataset shape: (25,000, 94)**
**Train: 17,500 | Val: 3,750 | Test: 3,750** (70/15/15 split, `random_state=42`)

---

## Model 1 — CBF + CF Hybrid Baseline

**Notebook:** `recommendation-paper-model-1.ipynb`

### Description

A classical hybrid combining **Content-Based Filtering (CBF)** and **User-Based Collaborative Filtering (CF)** via a weighted average:

```
prediction = α × CF_pred + (1 − α) × CBF_pred
```

- **CF:** cosine similarity on user–item pivot matrix (30×100); predicts from top-k similar users
- **CBF:** cosine similarity on article feature matrix (100×29); predicts from top-k similar items
- Final prediction clipped to [2.0, 5.0]

### Architecture

```
User-Item Matrix (30 × 100)
        │
   Cosine Similarity → User Similarity Matrix (30 × 30)
        │
   Top-k CF Prediction (k=5, weighted avg of neighbour ratings)
        │
        ├──── α=0.00 ────┐
                          ▼
Article Feature Matrix      Final Prediction  (clipped to [2.0, 5.0])
(100 × 29 content features)
        │
   Item Similarity (cosine, 100×100)
   Top-k CBF Prediction (k=5)
```

### Hyperparameter tuning

Alpha sweep (0.0 → 1.0, step 0.05) and k-sweep ([5,10,15,20,25,30,40,50]) run on validation set.

**Best found: α = 0.00 (pure CBF), k = 5**

> The alpha sweep found α=0.00 optimal — CF actually degraded performance. Content features alone were sufficient.

<!-- ADD PLOT HERE -->
<!-- File: Plots/alpha_sweep.png -->
<!-- What it shows: Validation MAE at each alpha value from 0.0 to 1.0 (step 0.05). The minimum sits at α=0.00, showing pure CBF is optimal. -->
![Alpha Sweep](Plots/alpha_sweep.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/k_sweep.png -->
<!-- What it shows: Validation MAE at each k value [5,10,15,20,25,30,40,50]. Best k=5. -->
![K-Neighbour Sweep](Plots/k_sweep.png)

### Results

| Component | MAE | MSE | RMSE |
|---|---|---|---|
| CF only (k=15, α=1.0) | 0.7613 | 0.8544 | 0.9244 |
| CBF only (k=15, α=0.0) | 0.3980 | 0.2383 | 0.4881 |
| Hybrid (k=15, α=0.4) | 0.5084 | 0.3560 | 0.5967 |
| **Final (k=5, α=0.00)** | **0.3554** | **0.2253** | **0.4746** |
| Paper baseline | 1.13 | 1.96 | 1.40 |

**Training time: 0:00:01**

<!-- ADD PLOT HERE -->
<!-- File: Plots/cbf_cf_results_improved.png -->
<!-- What it shows: 4-panel — predicted vs true scatter with regression fit, residuals histogram, model comparison bar chart (CF / CBF / Hybrid / Paper), error by rating bucket. -->
![Model 1 Results](Plots/cbf_cf_results_improved.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/per_rating_error.png -->
<!-- What it shows: MAE mean ± std per true rating value (2.0 to 5.0), plus count of test samples at each rating. -->
![Per Rating Error Model 1](Plots/per_rating_error.png)

**Per-rating-level MAE (Model 1):**

| True Rating | Mean MAE | Std | Count |
|---|---|---|---|
| 2.0 | 0.3215 | 0.2846 | 169 |
| 2.5 | 0.1672 | 0.1927 | 62 |
| 3.0 | 0.3470 | 0.2558 | 94 |
| 3.5 | 0.5099 | 0.3358 | 44 |
| 4.0 | 0.5677 | 0.3210 | 33 |
| 4.5 | 0.7336 | 0.4105 | 20 |
| 5.0 | 0.2416 | 0.3377 | 28 |

<!-- ADD PLOT HERE -->
<!-- File: Plots/summary_dashboard.png -->
<!-- What it shows: 3-panel — metrics comparison bar (ours vs paper), % improvement pie chart, hyperparameter summary table. -->
![Model 1 Summary Dashboard](Plots/summary_dashboard.png)

---

## Model 2 — CF Autoencoder

**Notebook:** `recommendation-paper-model-2.ipynb`

### Description

A **deep collaborative filtering autoencoder** that compresses a user's full rating vector (100 items) into a 64-dimensional latent space, then reconstructs it to predict all missing ratings. Uses **masked loss** — only rated entries contribute to backpropagation, ignoring the 37% of zeros.

### Architecture

```
Input: User Rating Vector (100,)
        │
   Input Dropout (15% noise regularisation)
        │
   ┌─── Encoder ───────────────────────────────────────────┐
   │  Dense(512) → LeakyReLU(0.1) → BatchNorm → Dropout(0.20)  │
   │  Dense(256) → LeakyReLU(0.1) → BatchNorm → Dropout(0.20)  │
   │  Dense(128) → LeakyReLU(0.1) → BatchNorm → Dropout(0.20)  │
   └───────────────────────────────────────────────────────┘
        │
   Latent: Dense(64, activation='tanh')    ← bottleneck
        │
   ┌─── Decoder ───────────────────────────────────────────┐
   │  Dense(128) → LeakyReLU(0.1) → BatchNorm → Dropout(0.14)  │
   │  Dense(256) → LeakyReLU(0.1) → BatchNorm → Dropout(0.14)  │
   │  Dense(512) → LeakyReLU(0.1) → BatchNorm → Dropout(0.14)  │
   └───────────────────────────────────────────────────────┘
        │
   Output: Dense(100, activation='linear')

Total parameters: 455,588 (1.74 MB)
Loss: Masked MSE  |  Metric: Masked MAE
Optimizer: Adam + Cosine Annealing LR (5-epoch warm-up, max 150 epochs)
```

**Key improvements over paper:**
- LeakyReLU(0.1) instead of ReLU — prevents dying neurons
- Label smoothing (α=0.05) on training targets
- Input dropout (15%) as noise regularisation
- Cosine annealing LR schedule with linear warm-up

### Hyperparameter tuning — 7 configurations

| Config | Encoder units | Latent | Dropout | LR | Val MAE |
|---|---|---|---|---|---|
| **C1 (best)** | **[512,256,128]** | **64** | **0.20** | **8e-4** | **0.0935** |
| C2 | [512,256,128] | 128 | 0.25 | 5e-4 | 0.1130 |
| C3 | [256,128,64] | 32 | 0.30 | 1e-3 | 0.1061 |
| C4 | [512,256] | 64 | 0.20 | 6e-4 | 0.1016 |
| C5 | [256,128] | 64 | 0.25 | 7e-4 | — |
| C6 | [512,256,128] | 64 | 0.30 | 3e-4 | — |
| C7 | [128,64,32] | 16 | 0.35 | 1e-3 | — |

<!-- ADD PLOT HERE -->
<!-- File: Plots/tuning_results.png -->
<!-- What it shows: 2-panel — bar chart of val MAE per config (best highlighted in gold), grouped bar of val MAE vs val loss across all 7 configs. -->
![Model 2 Hyperparameter Tuning](Plots/tuning_results.png)

### Training summary

| Metric | Value |
|---|---|
| Best config | enc=[512,256,128], latent=64, dropout=0.2, lr=8e-4, l2=1e-4 |
| Epochs ran | 50 |
| Training time | 0:00:15 |
| Best val MAE | 0.0985 |

### Results

| Metric | Our model | Paper baseline | Improvement |
|---|---|---|---|
| MAE | **0.8024** | 3.05 | 73.7% better |
| MSE | **0.8316** | 10.46 | 92.1% better |
| RMSE | **0.9119** | 3.23 | 71.8% better |

<!-- ADD PLOT HERE -->
<!-- File: Plots/autoencoder_results_improved.png -->
<!-- What it shows: 6-panel — training loss curve, masked MAE curve, cosine LR schedule with warm-up, predicted vs true scatter, residuals histogram, per-rating-bucket boxplot. -->
![Model 2 Results](Plots/autoencoder_results_improved.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/latent_space_pca.png -->
<!-- What it shows: 2D PCA projection of the 64-dim user latent vectors, coloured by average user rating. Clusters show the autoencoder groups similar users in latent space. -->
![Latent Space PCA](Plots/latent_space_pca.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/per_user_error_analysis.png -->
<!-- What it shows: 3-panel — MAE per user sorted best to worst, MAE vs number of ratings scatter, error boxplots by rating bucket. -->
![Per User Error Analysis](Plots/per_user_error_analysis.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/reconstruction_heatmap.png -->
<!-- What it shows: 3-panel heatmap — original user-item rating matrix (test sample), reconstructed matrix, absolute difference. Demonstrates visual reconstruction quality. -->
![Reconstruction Heatmap](Plots/reconstruction_heatmap.png)

---

## Model 3 — Dueling DQN + Double DQN + PER

**Notebook:** `recommendationpaper-model-3.ipynb`

### Description

Frames recommendation as a **sequential decision problem** using deep reinforcement learning. The agent receives a state (user feature vector + article feature vector = 60 dimensions) and selects which article to recommend from 500 possible actions. Ratings serve as rewards. Q-values are linearly scaled to [2.0, 5.0] for evaluation.

### Architecture — Dueling DQN

```
State Input: (60,)   ← user features (31) + item features (29)
        │
   Shared Encoder:
     Dense(256) → ReLU → BatchNorm → Dropout(0.25)
     Dense(128) → ReLU → BatchNorm → Dropout(0.25)
        │
        ├──────────────────────┬──────────────────────────
        │                      │
   Value stream:          Advantage stream:
   Dense(128) → ReLU      Dense(128) → ReLU
   Dense(1)               Dense(500)         ← 500 articles
        │                      │
        └──────────────────────┘
     Q(s,a) = V(s) + A(s,a) − mean(A(s,·))

Total parameters: 147,701 (576.96 KB)
Loss: Huber (delta=1.0)  |  Optimizer: Adam (lr=5e-4, clipnorm=1.0)
Replay buffer: Prioritised Experience Replay (α=0.6, β: 0.4→1.0)
```

### Key improvements over paper

| Hyperparameter | Original | Improved |
|---|---|---|
| Architecture | Vanilla DQN | Dueling DQN |
| Target update | Standard | Double DQN |
| Replay buffer | Uniform (2,000) | PER (5,000) |
| Loss function | MSE | Huber |
| L2 regularisation | None | 1e-4 |
| Gamma (γ) | 0.95 | 0.97 |
| Epsilon decay | 0.995 | 0.997 |
| Batch size | 128 | 64 |
| Target sync | 100 steps | 50 steps |
| Episodes | 3 | 5 |
| Reward shaping | None | +0.15 preference-match bonus |

### Training dynamics

| Episode | Reward | Huber Loss | Val MAE | Epsilon |
|---|---|---|---|---|
| 1 | 8,809.5 | 35.9631 | 1.5264 | 0.050 |
| 2 | 8,763.5 | 56.6317 | 1.4342 | 0.050 |
| 3 | 8,871.5 | 45.0494 | 1.6114 | 0.050 |
| 4 | 8,864.0 | 16.3016 | 1.4429 | 0.050 |
| 5 | 8,832.0 | 15.3274 | 1.3543 | 0.050 |

**Total steps: 15,000 | Training time: 0:13:35 | Final epsilon: 0.05**

<!-- ADD PLOT HERE -->
<!-- File: Plots/dqn_training_dynamics.png -->
<!-- What it shows: 4-panel — episode cumulative rewards, Huber loss curve (raw + 60-step smoothed), validation MAE per episode with paper baseline line, epsilon decay schedule across all steps. -->
![DQN Training Dynamics](Plots/dqn_training_dynamics.png)

### Results

| Metric | Our model | Paper baseline | Improvement |
|---|---|---|---|
| MAE | **1.3448** | 3.22 | 58.2% better |
| MSE | **2.6515** | 12.22 | 78.3% better |
| RMSE | **1.6283** | 3.49 | 53.3% better |

**Per-rating-level MAE (Model 3):**

| True Rating | Mean MAE | Std | Count |
|---|---|---|---|
| 2.0 | 1.6219 | 1.2034 | 1,227 |
| 2.5 | 1.3112 | 0.8420 | 351 |
| 3.0 | 1.1868 | 0.6375 | 1,127 |
| 3.5 | 1.1684 | 0.2549 | 336 |
| 4.0 | 1.1352 | 0.5185 | 325 |
| 4.5 | 1.2348 | 0.9599 | 242 |
| 5.0 | 1.3727 | 1.1859 | 142 |

<!-- ADD PLOT HERE -->
<!-- File: Plots/dqn_prediction_quality.png -->
<!-- What it shows: 4-panel — predicted vs true scatter with regression fit, residuals histogram with mean error line, residuals vs predicted (rolling mean), Q-value spread distribution (higher spread = more decisive agent). -->
![DQN Prediction Quality](Plots/dqn_prediction_quality.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/dqn_vs_paper.png -->
<!-- What it shows: 3-panel bar chart comparing improved DQN vs paper DQN on MAE, RMSE, and MSE. -->
![DQN vs Paper](Plots/dqn_vs_paper.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/dqn_per_rating_error.png -->
<!-- What it shows: 2-panel — bar chart of mean MAE ± std per true rating value, bar chart of test sample count per rating value. -->
![DQN Per Rating Error](Plots/dqn_per_rating_error.png)

<!-- ADD PLOT HERE -->
<!-- File: Plots/dqn_summary_dashboard.png -->
<!-- What it shows: 3-panel — metrics comparison bar (ours vs paper), % improvement pie chart, full hyperparameter table comparing original vs improved settings. -->
![DQN Summary Dashboard](Plots/dqn_summary_dashboard.png)

---

## Model 4 — Hybrid Multi-Tower Neural Network

**Notebook:** `recommendation-paper-model-4.ipynb`

### Description

A **four-tower neural network** that processes four distinct feature groups through separate specialised sub-networks, compressing each into a low-dimensional embedding before fusing them into a single rating prediction. This is the paper's highest-performing architecture.

### Architecture

```
User Features (31,)    Item Features (36,)    Time Features (5,)    Problem Features (12,)
      │                       │                      │                       │
 User Tower:            Item Tower:            Time Tower:           Problem Tower:
 Dense(128)→ReLU        Dense(256)→ReLU        Dense(16)→ReLU        Dense(32)→ReLU
 BatchNorm              BatchNorm              BatchNorm             BatchNorm
 Dropout(0.2)           Dropout(0.2)           Dropout(0.2)          Dropout(0.2)
 Dense(16)→ReLU         Dense(16)→ReLU                               Dense(32)→ReLU
 [embed: 16]            [embed: 16]            [out: 16]             [out: 32]
      │                       │                      │                       │
      └───────────────────────┴──────────────────────┴───────────────────────┘
                                       │
                           Concatenate [16+16+16+32 = 80]
                                       │
                           Dense(128) → ReLU → BatchNorm → Dropout(0.2)
                                       │
                           Dense(32)  → ReLU
                                       │
                           Dense(1)   → Linear (rating prediction)

Total parameters: 38,081 (148.75 KB)
Loss: MSE  |  Metric: MAE  |  Optimizer: Adam
Hardware: 2× Tesla T4 GPU
```

**Feature groups:**
- **User Tower (31):** 18 preference scores + 7 user-type one-hots + 6 pref. presentation-type one-hots
- **Item Tower (36):** 18 quality dimension scores + 6 pres.-type one-hots + 5 audience-type one-hots + item stats
- **Time Tower (5):** hour_sin, hour_cos, day_sin, day_cos, hours_since_first
- **Problem Tower (12):** 12 exponentially decay-weighted problem dimension scores (λ=0.01)

### Hyperparameter tuning — 5 configurations

| Config | User units | Item units | Dropout | LR | Val MAE |
|---|---|---|---|---|---|
| C1 | 128 | 256 | 0.3 | 5.17e-4 | — |
| **C2 (best)** | **128** | **256** | **0.2** | **1e-3** | **0.0865** |
| C3 | 64 | 256 | 0.3 | 5e-4 | — |
| C4 | 128 | 128 | 0.2 | 1e-3 | — |
| C5 | 128 | 256 | 0.4 | 5e-4 | — |

### Training summary

| Metric | Value |
|---|---|
| Best config | user=128, item=256, dropout=0.2, lr=0.001 |
| Epochs ran | 200 |
| Training time | 0:01:38 |
| Best train loss | 0.0335 |
| Best val loss | 0.0257 |
| Best train MAE | 0.1158 |
| Best val MAE | 0.0865 |

### Results

| Metric | Our model | Paper baseline | Improvement |
|---|---|---|---|
| MAE | **0.0803** | 0.11 | 27.0% better |
| MSE | **0.0254** | 0.033 | 23.0% better |
| RMSE | **0.1595** | 0.18 | 11.4% better |

<!-- ADD PLOT HERE -->
<!-- File: Plots/hybrid_results.png -->
<!-- What it shows: 6-panel — training loss curve (MSE), MAE curve, tuning config comparison bar (gold = best), predicted vs true scatter, error distribution histogram, paper vs ours bar comparison on MAE/RMSE/MSE. -->
![Model 4 Results](Plots/hybrid_results.png)

---

## Model 5 — Hybrid + Custom Pre-Trained Embeddings

**Notebook:** `recommendation-paper-model-5.ipynb`

### Description

Extends Model 4 with a **two-stage training pipeline**. First, two small autoencoders learn compact 16-dimensional embeddings from user and item feature matrices. These embeddings replace the raw inputs, and the combined vector (16 + 16 + 5 + 12 = **49 dims**) feeds a prediction head.

### Architecture

**Stage 1 — Pre-train embedding autoencoders (separately for user and item):**

```
User Autoencoder (31 → 16 → 31):          Item Autoencoder (36 → 16 → 36):
Input(31) → Dense(64) → ReLU              Input(36) → Dense(64) → ReLU
→ BatchNorm → Dropout(0.2)                → BatchNorm → Dropout(0.2)
→ Dense(16, ReLU)  [EMBED bottleneck]     → Dense(16, ReLU)  [EMBED bottleneck]
→ Dense(64) → ReLU → Output(31)           → Dense(64) → ReLU → Output(36)
```

**Stage 2 — Prediction head on combined embeddings:**

```
[User Embed(16)] + [Item Embed(16)] + [Time(5)] + [Problem(12)]
                              │
                    Combined Input (49,)
                              │
               Dense(256) → ReLU → BatchNorm → Dropout(0.3)
               Dense(128) → ReLU → BatchNorm → Dropout(0.3)
               Dense(64)  → ReLU → BatchNorm → Dropout(0.3)
                              │
               Dense(1) → Linear (rating prediction)

Total parameters: 17,665 (69.00 KB)
Loss: MSE  |  Metric: MAE  |  Optimizer: Adam
```

### Hyperparameter tuning — 5 configurations

| Config | Units | Dropout | LR |
|---|---|---|---|
| C1 | [128,64,32] | 0.3 | 1e-3 |
| C2 | [128,64,32] | 0.2 | 5e-4 |
| **C3 (best)** | **[256,128,64]** | **0.3** | **1e-3** |
| C4 | [64,32] | 0.2 | 1e-3 |
| C5 | [128,64] | 0.3 | 5e-4 |

### Training summary

| Metric | Value |
|---|---|
| Best config | units=[256,128,64], dropout=0.3, lr=0.001 |
| Epochs ran | 123 |
| Training time | 0:00:45 |
| Best train loss | 0.2835 |
| Best val loss | 0.2463 |
| Best train MAE | 0.3935 |
| Best val MAE | 0.3266 |

### Results

| Metric | Our model | Paper baseline | Improvement |
|---|---|---|---|
| MAE | **0.3408** | 0.49 | 30.4% better |
| MSE | **0.2496** | 0.36 | 30.7% better |
| RMSE | **0.4996** | 0.60 | 16.7% better |

> **Key finding:** Model 5 (MAE = 0.3408) is significantly worse than Model 4 (MAE = 0.0803) despite being an extension of it. The 16-dim pre-trained embeddings compressed away useful information that the raw 31/36-dim features preserved. Dimensionality reduction via autoencoder does not always improve downstream performance when original features are already well-structured.

<!-- ADD PLOT HERE -->
<!-- File: Plots/hybrid_emb_results.png -->
<!-- What it shows: 6-panel — training loss curve, MAE curve, tuning config bar (gold = best), predicted vs true scatter, error distribution, paper vs ours comparison bar on MAE/RMSE/MSE. -->
![Model 5 Results](Plots/hybrid_emb_results.png)

---

## Results Summary

| Model | Architecture | Test MAE | Test MSE | Test RMSE | Paper MAE | Paper RMSE | Beats Paper? |
|---|---|---|---|---|---|---|---|
| Model 1 | CBF + CF Hybrid | **0.3554** | 0.2253 | 0.4746 | 1.13 | 1.40 | ✅ YES |
| Model 2 | CF Autoencoder | **0.8024** | 0.8316 | 0.9119 | 3.05 | 3.23 | ✅ YES |
| Model 3 | Dueling DQN + PER | **1.3448** | 2.6515 | 1.6283 | 3.22 | 3.49 | ✅ YES |
| Model 4 | Hybrid Multi-Tower NN | **0.0803** | 0.0254 | 0.1595 | 0.11 | 0.18 | ✅ YES |
| Model 5 | Hybrid + Embeddings | **0.3408** | 0.2496 | 0.4996 | 0.49 | 0.60 | ✅ YES |

**Every single model beats the paper baseline.**

---

## All Models Comparison

<!-- ADD PLOT HERE -->
<!-- File: Plots/all_models_comparison.png -->
<!-- What it shows: 3-panel bar chart — one panel each for MAE, RMSE, and MSE — comparing all 5 models side by side including paper baselines. Generated by Model 2 notebook. -->
![All Models Comparison](Plots/all_models_comparison.png)

### Key findings

| Finding | Detail |
|---|---|
| Best model | Model 4 (MAE = 0.0803) — four-tower fusion architecture |
| Largest relative improvement | Model 3: DQN MAE improved by **58.2%** (3.22 → 1.3448) |
| Pure CBF beats hybrid | Model 1: optimal α = 0.00 — content features alone outperformed CF blend |
| Embedding compression hurt | Model 5 (0.3408) worse than Model 4 (0.0803) — 16-dim bottleneck lost information |
| RL highest absolute MAE | Model 3 MAE = 1.3448 — Q-values are not naturally calibrated to rating scale |

---

## Requirements

```
Python        >= 3.8
TensorFlow    >= 2.19.0
NumPy         >= 1.21
Pandas        >= 1.3
scikit-learn  >= 1.0
Matplotlib    >= 3.4
Seaborn       >= 0.11
tqdm          >= 4.62
```

GPU recommended for Models 3, 4, 5. Tested on 2× Tesla T4 (Kaggle environment, TensorFlow 2.19.0).

---

## Author

Janvi Porwal
BTech AI & DS

```

---

## License

Released for academic and research purposes. Refer to the Kaggle dataset page for dataset-specific licensing terms.
