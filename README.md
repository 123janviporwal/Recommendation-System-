# Presentation Advisor Recommendation System

A research project implementing and benchmarking five recommendation system architectures for presentation quality advising. Each model is reproduced from a reference paper and improved upon, evaluated on the **Presentation Advisor Dataset** from Kaggle.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Feature Engineering](#feature-engineering)
- [Plots & Visualisations](#plots--visualisations)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Project Overview

This project explores five distinct recommendation approaches — from classical collaborative filtering to deep reinforcement learning — all applied to the task of recommending presentation improvement advice to users. Each notebook independently loads the data, performs exploratory data analysis (EDA), engineers features, and trains/evaluates an improved model against the paper's reported baselines.

**Goals:**
- Reproduce the baselines reported in the reference paper.
- Apply systematic improvements (architecture, regularisation, hyperparameter tuning).
- Compare all five models on MAE, MSE, and RMSE on a held-out test set.

---

## Dataset

**Source:** [Presentation Advisor Data — Kaggle](https://www.kaggle.com/datasets/janvijain96/presentation-advisor-data)

| File | Description |
|---|---|
| `interaction_data.csv` | User–article interactions with ratings and feature vectors |
| `ratings_matrix.csv` | Pivot-style ratings matrix |
| `presentations.csv` | Raw presentation metadata |
| `presentations_df.csv` | Processed presentation features |
| `recommendations.csv` | Article-level recommendation metadata (presentation type, audience type, main issue, popularity) |
| `user_profiles.csv` | User type, location, preferences, preferred presentation type |

**Key statistics:**
- Ratings range: 2.0 – 5.0
- User types: business, teacher, student, researcher, manager, technical, specialist
- Presentation types: formal, creative, business, educational, technical, persuasive
- Audience types: academic, business, technical, kids, general
- 18 presentation quality dimensions (e.g. Graphics, Readability, Bullet points, Infographics)

All models use a **70/15/15 train/validation/test split** (stratified by `random_state=42`).

---

## Models

### Model 1 — CBF + CF Hybrid Baseline (`recommendation-paper-model-1.ipynb`)

A classical hybrid combining **Content-Based Filtering (CBF)** and **Collaborative Filtering (CF)** via cosine similarity.

**Approach:**
- User–item cosine similarity matrix built from a pivot table of ratings.
- Item–item cosine similarity matrix built from presentation feature vectors.
- Final prediction: `α × CF_pred + (1 − α) × CBF_pred`, clipped to [2, 5].

**Key improvements over paper:**
- Alpha sweep on validation set to find optimal blend weight (best α ≈ 0.40, more weight to CBF).
- K-neighbour sweep to find optimal k (best k = 15, up from 10).
- Per-rating-level error analysis.

**Paper baseline:** MAE = 1.13, MSE = 1.96, RMSE = 1.40

---

### Model 2 — CF Autoencoder (`recommendation-paper-model-2.ipynb`)

A **deep collaborative filtering autoencoder** that learns a compressed latent representation of the user–rating vector and reconstructs missing ratings.

**Architecture:**
- Encoder: Dense(512) → Dense(256) → Dense(128) → Latent(64), with LeakyReLU, BatchNorm, Dropout.
- Decoder: mirrors encoder (reversed), with reduced dropout.
- Loss: masked MSE (ignores unrated items); masked MAE as metric.

**Key improvements over paper:**
- LeakyReLU activations instead of ReLU.
- Label smoothing (α = 0.05) applied to training targets.
- Input noise dropout (15%) for regularisation.
- Cosine annealing learning rate schedule with linear warm-up (5 epochs).
- Expanded hyperparameter grid search across 7 configurations.
- Per-user and per-rating-bucket error analysis.

**Paper baseline:** MAE = 3.05, MSE = 10.46, RMSE = 3.23

---

### Model 3 — Dueling DQN + Double DQN + PER (`recommendationpaper-model-3.ipynb`)

A **deep reinforcement learning** approach framing recommendation as a sequential decision problem. The agent selects articles to recommend and receives rating-based rewards.

**Architecture — Dueling DQN:**
- Shared encoder: Dense(256) → BN → Dropout → Dense(128) → BN → Dropout.
- Value stream: Dense(128) → Dense(1).
- Advantage stream: Dense(128) → Dense(action_dim).
- Q(s, a) = V(s) + A(s, a) − mean(A(s, ·)).

**State:** concatenation of user feature vector and article feature vector.

**Key improvements over paper:**

| Hyperparameter | Original | Improved |
|---|---|---|
| Architecture | Vanilla DQN | Dueling DQN |
| Target update | Standard | Double DQN |
| Replay buffer | Uniform (2000) | Prioritised Experience Replay (5000) |
| Loss function | MSE | Huber |
| L2 regularisation | None | 1e-4 |
| Gamma | 0.95 | 0.97 |
| Epsilon decay | 0.995 | 0.997 |
| Batch size | 128 | 64 |
| Target sync | 100 steps | 50 steps |
| Episodes | 3 | 5 |
| Reward shaping | — | Preference-match bonus (+0.15) |

**Paper baseline:** MAE = 3.22, MSE = 12.22, RMSE = 3.49

---

### Model 4 — Hybrid Multi-Tower Neural Network (`recommendation-paper-model-4.ipynb`)

A **four-tower neural network** that processes user features, item features, temporal signals, and problem sequence features through separate towers before fusing them.

**Architecture (from paper Table 5):**

| Tower | Input | Architecture |
|---|---|---|
| User Tower | User preference + type + pres. type features | Dense(128) → BN → Dropout → Embed(16) |
| Item Tower | Item quality features + pres./audience type | Dense(256) → BN → Dropout → Embed(16) |
| Time Tower | Cyclic time features + decay weight | Dense(16) → BN → Dropout |
| Problem Tower | Exponentially-weighted problem dimensions | Dense(32) → BN → Dropout → Proj(32) |

**Fusion:** Concatenate all tower outputs → Dense(128) → BN → Dropout → Dense(32) → Output (linear).

**Temporal features:** hour/day/month encoded as sine/cosine pairs; hours since first interaction; exponential decay weight (λ = 0.01) applied to problem dimension scores.

**Training:** Adam (lr = 5.17e-4), EarlyStopping + ReduceLROnPlateau + ModelCheckpoint, hyperparameter tuning across multiple configurations.

**Paper baseline:** MAE = 0.11, MSE = 0.033, RMSE = 0.18

---

### Model 5 — Hybrid Multi-Tower + Custom Pre-Trained Embeddings (`recommendation-paper-model-5.ipynb`)

An extension of Model 4 that augments the multi-tower architecture with **custom pre-trained embeddings** learned from the dataset, replacing raw feature inputs with richer representations.

**Key additions over Model 4:**
- Pre-trained user and item embedding layers trained as a separate unsupervised step.
- Embeddings are frozen or fine-tuned during the main training phase.
- Concatenated embedding features fed into the same multi-tower fusion architecture.
- Same cyclic time features and exponentially-decayed problem sequence features as Model 4.

**Paper baseline:** MAE = 0.49, MSE = 0.36, RMSE = 0.60

---

## Results Summary

| Model                     | MAE     | MSE     | RMSE    | Paper MAE | Beats Paper |
|--------------------------|--------|--------|--------|-----------|-------------|
| CBF + CF (Hybrid)        | 0.5255 | 0.3908 | 0.6252 | 1.13      |  Yes       |
| Autoencoder              | 0.7861 | 0.7964 | 0.8924 | 3.05      |  Yes       |
| DQN                      | 1.4968 | 3.3036 | 1.8176 | 3.22      |  Yes       |
| Hybrid Multi-Tower       | 0.0734 | 0.0259 | 0.1611 | 0.11      |  Yes       |
| Hybrid + Embeddings      | 0.3324 | 0.2514 | 0.5014 | 0.49      |  Yes       |


---

## Repository Structure

```
.
├── recommendation-paper-model-1.ipynb   # CBF + CF Hybrid Baseline
├── recommendation-paper-model-2.ipynb   # CF Autoencoder
├── recommendationpaper-model-3.ipynb    # Dueling DQN + Double DQN + PER
├── recommendation-paper-model-4.ipynb   # Hybrid Multi-Tower Neural Network
├── recommendation-paper-model-5.ipynb   # Hybrid + Custom Pre-Trained Embeddings
├── Plots/                               # Generated visualisation outputs
└── README.md
```

Each notebook is self-contained and follows the same structure:
1. **Imports & Setup** — seeds, GPU config, output directories.
2. **Data Loading** — reads all six CSV files from the dataset.
3. **EDA** — ratings distribution, user/item statistics, temporal patterns, preference analysis.
4. **Feature Engineering** — cyclic encoding, user/item stats, decay weights, scaling.
5. **Hyperparameter Tuning** — grid search or sweep on validation set.
6. **Model Training** — final model with callbacks.
7. **Test Evaluation** — MAE, MSE, RMSE with comparison to paper.
8. **Visualisation** — multi-panel result plots saved to `saved_models/`.
9. **Save** — model weights (`.keras`) and metrics (`.json`).

---

## Installation & Usage

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

Download from Kaggle and place the six CSV files under a directory matching the path in each notebook:

```
/kaggle/input/datasets/<username>/presentation-advisor-data/
    interaction_data.csv
    presentations.csv
    presentations_df.csv
    ratings_matrix.csv
    recommendations.csv
    user_profiles.csv
```

If running locally, update the file paths at the top of the data loading cell in each notebook.

### 4. Run the notebooks

Run each notebook independently. They do not depend on each other (Models 4 and 5 share feature engineering code but Model 5 should be run after Model 4 since it reuses the train/val/test split indices).

```bash
jupyter notebook recommendation-paper-model-1.ipynb
```

Or run on **Kaggle** directly — all notebooks are set up for the Kaggle environment with GPU support.

---

## Feature Engineering

All models share a common feature engineering pipeline (implemented independently in each notebook):

**Temporal features:**
- Hour of day, day of week, month — encoded as sine/cosine pairs to preserve cyclicality.
- Hours/days since first user interaction.
- Exponential decay weight: `exp(−λ × days_since_first)` applied to problem dimension scores (λ = 0.005 in Model 3, λ = 0.01 in Models 4/5).

**User statistics:** per-user mean rating, rating count, rating standard deviation.

**Item statistics:** per-item mean rating, rating count, rating standard deviation.

**Preference match score** (Model 3): counts alignment between a user's stated preference scores and the article's issue scores.

**Normalisation:** all continuous features scaled to [0, 1] with MinMaxScaler fitted on the training set.

---

## Plots & Visualisations

Each notebook generates and saves the following plots:

**EDA plots (all models):**
- Rating distribution (interaction data + ratings matrix)
- Ratings per user and ratings per article
- User preference scores and correlation heatmap
- User type, presentation type, audience type distributions
- Average rating by segment (audience type, preferred presentation type)
- Temporal patterns: monthly volume, average rating over time

**Model-specific plots:**
- Model 1: alpha sweep, k-neighbour sweep, predicted vs true, residuals, per-rating error.
- Model 2: loss/MAE curves, cosine LR schedule, per-user error analysis, autoencoder reconstruction heatmap.
- Model 3: episode rewards, training loss (Huber), validation MAE per episode, epsilon decay, Q-value spread, DQN vs paper comparison.
- Models 4/5: tuning bar charts, loss/MAE curves, predicted vs true scatter, error distribution, paper vs ours comparison.

---

## Requirements

```
Python        >= 3.8
TensorFlow    >= 2.10
NumPy         >= 1.21
Pandas        >= 1.3
scikit-learn  >= 1.0
Matplotlib    >= 3.4
Seaborn       >= 0.11
tqdm          >= 4.62
```

GPU recommended for Models 3, 4, and 5. All notebooks include automatic GPU memory growth configuration via `tf.config.experimental.set_memory_growth`.

---

## Citation

If you use this code or the dataset in your research, please cite the original paper and dataset:

```bibtex
@dataset{presentation_advisor_data,
  author  = {Janvi Porwal},
  title   = {Presentation Advisor Data},
}
```

---

## License

This project is released for academic and research purposes. Please refer to the Kaggle dataset page for dataset-specific licensing terms.
