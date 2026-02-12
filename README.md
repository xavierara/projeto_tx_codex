# RL Pricing for Used-Car Listings

This repository implements a **model-based reinforcement learning pipeline** for dynamic pricing of used-car listings. It trains a tabular Q-learning agent to optimise pricing decisions using **empirical transitions** learned from real seller behaviour (not ML-predicted probabilities).

## Repository Structure

```
├── data/
│   └── autos.csv                          # German used-car listings dataset (~370k records)
├── pricing_rl.ipynb                       # Main pipeline notebook
└── README.md                              # This file
```

## Main Notebook: `pricing_rl.ipynb`

**Structure** (56 cells):

1. **Cell 0** — Config + Imports (hyperparameters, translations, LightGBM settings)
2. **Cell 1** — Load + Clean Data (parse dates, duration filtering)
3. **Cell 2** — Build Ad-ID (identify repeated listings for trajectory extraction)
4. **Cell 3** — German→English Translation (standardize categorical values)
5. **Cell 4** — Outlier Filtering (price, age, mileage, power)
6. **Cell 5** — Segmentation (group cars into ~1000 vehicle segments)
7. **Cell 6** — Train-Validation Split (80/20 temporal split)
8. **Cell 7** — Price Model Training (LightGBM on log-price with features)
9. **Cell 8** — Price Deviation (δ = log(price/p₀), clipped to [-0.7, 0.7])
10. **Cell 9** — Empirical Transition Table (P(sale | segment, week, action) from training data only—**no data leakage**)
11. **Cell 10** — Delta Bounds (per-segment clipping limits)
12. **Cell 11** — RL Environment (`CarPricingEnv` class with empirical transitions)
13. **Cell 12** — Q-Learning Training (tabular Q-learning, 10k episodes, ε-greedy)
14. **Cell 13** — Baselines & Evaluation (keep_price vs RL agent on validation set)
15. **Cells 14–26** — Comprehensive Diagnostics
    - 9.1: Price model residuals (histogram, QQ-plot, predictions)
    - 9.2: Sensitivity analysis (holding cost sweep)
    - 9.3: Segment-level performance breakdown
    - 9.4: Price action distribution over time
    - 9.5: Event probability calibration (reliability diagram)
    - 9.6: Q-value heatmaps & greedy action maps
    - 9.7: Out-of-distribution state detection
    - 9.8: Q-learning convergence & stability
16. **Cells 27–31** — Exploratory Examples
    - 10.1: Segment composition examples
    - 10.2: Price prediction samples
    - 10.3: Policy comparison on identical cars (baseline vs RL)

See [DATA.md](DATA.md) for detailed preprocessing description.

## Prerequisites

- **Python 3.8+**
- **Jupyter Notebook** or **VS Code** with Jupyter extension

## Setup

### 1. Clone or download the project

```bash
cd projeto_tx_codex
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib lightgbm scikit-learn scipy
```

### 4. Verify the dataset

Ensure `data/autos.csv` exists. The notebook expects:
```python
DATA_PATH = "data/autos.csv"
```

## Configuration

All hyperparameters are in **Cell 1** of `pricing_rl.ipynb`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TMAX` | 12 | Maximum weeks per listing |
| `ACTION_PCTS` | `[-5%, -3%, 0%, +3%, +5%]` | Discrete price adjustments |
| `ALPHA` | 0.10 | Q-learning learning rate |
| `GAMMA` | 0.98 | Discount factor |
| `HOLDING_COST` | €40.0 | Weekly cost of unsold listing |
| `N_EPISODES` | 10,000 | RL training episodes |
| `N_EVAL` | 1,000 | Evaluation episodes |
| `EPS_START / EPS_END / EPS_DECAY` | 1.0 / 0.05 / 0.995 | Exploration schedule |
| `DELTA_BINS` | 16 bins | Price deviation discretisation ([-0.5, 0.5]) |
| `MIN_MODEL_COUNT` | 200 | Min listings per brand-model to retain |

## Key Concepts

### RL Framework

- **State**: $(s_{\text{seg}}, t_{\text{week}}, \delta_{\text{bucket}})$
  - Segment ID (~1000 unique values)
  - Week number (1–12)
  - Price deviation bin (16 buckets)

- **Action**: Discrete price adjustment from -5\%, -3\%, 0\%, +3\%, +5\%

- **Reward**: 
  - Sale: receive actual selling price (€)
  - No sale: -€40 weekly holding cost

- **Transition**: P(sale | segment, week, action)

*Alternative (not used)*: Learned hazard model h(x,t,Δ) — more flexible but requires more careful calibration.

### Train-test split

1. **Train-validation split** (80/20 temporal -- train on older ads)
2. **Empirical transitions** extracted from **training data only**
3. **Evaluation** on **validation data** (clean holdout)
4. **Price model** trained on training data, used for all

Result: RL agent learns from training market, tested on unseen validation market.

## Output Interpretation

### Evaluation Metrics (from Cell 14)

| Metric | Interpretation |
|--------|---|
| `avg_return` (€) | Expected revenue per episode (episode = listing lifespan) |
| `avg_weeks` (weeks) | Mean time to sale |
| `survival curve` | P(not sold by week t) — how long listings stay active |
| `avg_price_ratio_cond_event` | Mean price multiplier when sale occurs |

