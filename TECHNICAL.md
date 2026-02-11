# RL Pricing for Used-Car Listings

This repository implements a **model-based reinforcement learning pipeline** for dynamic pricing of used-car listings. The system extracts empirical transition probabilities from real price trajectories and trains a tabular Q-learning agent in a simulated environment to optimize pricing decisions, maximizing expected revenue while accounting for holding costs.

**Main Notebook**: [rl_pricing_model_based_id.ipynb](rl_pricing_model_based_id.ipynb) — Recommended for production use (clean train/validation split, no data leakage)

## Overview

The pipeline operates in two phases:

1. **Offline Learning Phase**: Build predictive models (price & hazard) from historical data, then train an RL agent in a simulated environment
2. **Deployment Phase**: Use the learned Q-table to recommend prices for new listings

---

## Pipeline Walkthrough

### 1. Data Loading & Cleaning

**Purpose**: Prepare raw `autos.csv` for downstream modeling

**Steps**:
- Load German used-car listings dataset
- Parse datetime columns: `dateCreated`, `dateCrawled`, `lastSeen`
- Drop records with missing dates
- Calculate listing duration: `duration_days = (lastSeen - dateCreated)`
- Translate German categorical values to English (e.g., `Benzin` → `Petrol`)
- Filter extreme outliers (e.g., prices > 99th percentile)

**Output**: Clean DataFrame with temporal features and standardized labels

---

### 2. Segmentation: Composite Grouping

**Purpose**: Partition cars into homogeneous segments for modeling

**Strategy**:
```
Segment = (Brand, Model, Vehicle Type, Year Bin)
```

**Details**:
- Extract brand-model pairs; keep only those with ≥200 listings (`MIN_MODEL_COUNT`)
- Vehicle type: `Sedan`, `SUV`, `Estate`, `Coupe`, `Convertible`, `Van`, `City Car`, `Off-road`, `Pickup`
- Year bins: 5-year ranges (e.g., 2020–2024, 2015–2019)
- Assign each listing a `segment_id`

**Rationale**: Ensures sufficient data per segment for reliable empirical transition estimates

---

### 3. Baseline Price Model: $p_0(x)$

**Purpose**: Predict fair-market price for a car given its features

**Model**: LightGBM regressor on log-transformed prices

**Features**:
- Numerical: `mileageKm`, `powerPS`, `year`, `monthOfRegistration`
- Categorical: `gearbox`, `fuelType`, `seller`

**Training**:
```python
# Fit on log-price target
y = log(price)

lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8
).fit(X_train, y_train)
```

**Output**: Function $p_0(x)$ that predicts baseline price in €

**Usage**: Represents the "fair" price; deviations from $p_0$ are encoded as state delta $\Delta$

---

### 4. Ad-ID Construction

**Purpose**: Identify repeated listings of the same car across multiple weekly crawls

**Method**: Composite key
```
ad_id = (mileageKm, brand, model, year, powerPS, gearbox, fuelType, seller)
```

**Why**: Enables extraction of real price trajectories (how sellers reprice over time)

**Output**: `df_sorted` with ads grouped by ID and sorted by `dateCrawled`

---

### 5. Empirical Transition Table

**Purpose**: Learn empirical probabilities $P(\text{sale} \mid \text{segment}, \text{week}, \text{action})$ from real data

**Data Source**: **Training set only** (no data leakage into evaluation)

**Process**:

1. **Extract trajectories**: For each ad-id, collect price sequence across weeks
   ```
   Trajectory: w1 → w2 → w3 → ... → sold/disappeared
   ```

2. **Compute price deltas**: For each transition
   ```
   Δ = log(price_t) - log(p0(x))
   action_bin = argmin_a |price_t - p0(x) * exp(ACTION_PCTS[a])|
   ```

3. **Aggregate events**:
   ```
   For each (segment_id, week_t, action_bin):
       P(sale | seg, t, a) = [# sales in next week] / [# listings]
   ```

4. **Result**: DataFrame with columns:
   - `seg`: segment ID
   - `t_week`: week (1–12)
   - `action_bin`: action index (0–4 for 5 actions)
   - `sold_next_obs`: empirical sale probability

**Key Property**: Clean train/valid split ensures unbiased evaluation

---



## Reinforcement Learning Framework

The RL agent learns a **policy** $\pi(s) \rightarrow a$ that maps states to pricing actions to maximize cumulative discounted reward.

### State Space: $s = (\text{seg\_id}, t, \Delta)$

| Component | Description | Values |
|-----------|-------------|--------|
| `seg_id` | Segment ID | 1–200+ |
| `t` | Weeks in listing | 1–12 |
| `delta_bucket` | Price deviation bin | 16 bins in $[-0.5, 0.5]$ |

**Discretization**: Delta is binned into 16 equal intervals; e.g., $\Delta \in [-0.03, 0)$ maps to bucket 5.

**Intuition**: State captures *what car* (segment), *how long* it's been listed, and *how far* the current price deviates from fair market value.

---

### Action Space: $a \in \{0, 1, 2, 3, 4\}$

Five discrete price adjustments:

| Action | Adjustment | Effect |
|--------|------------|--------|
| 0 | −5% | Aggressive markdown |
| 1 | −3% | Moderate discount |
| 2 | 0% | Hold price (do nothing) |
| 3 | +3% | Small increase (risk) |
| 4 | +5% | Large increase (high risk) |

**Mathematical Form**:
```
action_pct ∈ ACTION_PCTS = [-0.05, -0.03, 0.0, +0.03, +0.05]
new_price = p0(x) * exp(delta + action_pct)
```

---

### Reward Signal: $r_t$

**Immediate Reward** (received each step):

```python
if sale_occurs:
    r_t = selling_price  # Revenue in €
else:
    r_t = -HOLDING_COST  # Weekly opportunity cost (€40)
```

**Intuition**: 
- Selling generates revenue (positive)
- Holding unsold listings costs money (negative)
- Agent must balance **patience** (wait for buyer at high price) vs. **liquidation risk** (reduce price early)

**Cumulative Objective**:
$$V(s) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0 = s, \pi\right]$$

where $\gamma = 0.98$ (discount factor).

---

### RL Environment: CarPricingEnv

**Dynamics** (each step):

1. **Current state**: $(x, t, \Delta)$
2. **Agent chooses action**: $a \sim \pi(s)$
3. **Update price**:
   ```
   Δ_new = clip(Δ + ACTION_PCTS[a])
   price_new = p0(x) * exp(Δ_new)
   ```
4. **Sample sale event** (from empirical transitions):
   ```
   p_sale = P(sale | segment, week, action)  [from real data]
   sale ← Bernoulli(p_sale)
   ```
5. **Transition**:
   - **If sale**: $r = \text{price\_new}$, $\text{done} = \text{True}$
   - **Else**: $r = -\text{HOLDING\_COST}$, transition to $(x, t+1, \Delta_\text{new})$

**Termination**: Episode ends when sale occurs or $t > \text{TMAX} = 12$ weeks.

**Key Property**: Sale probabilities are grounded in **observed seller behavior**, not ML predictions, ensuring realistic training dynamics.

---

### Q-Learning: Policy Training

**Algorithm**: Tabular Q-learning with ε-greedy exploration

**Q-Table**: Maps states to action values
```python
Q[(seg_id, t, delta_bucket)] = [Q_a0, Q_a1, Q_a2, Q_a3, Q_a4]
```

**Update Rule** (Bellman temporal-difference):
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

where:
- $\alpha = 0.10$ (learning rate)
- $\gamma = 0.98$ (discount factor)
- $r + \gamma \max_{a'} Q(s', a')$ is the **TD target**
- $Q(s, a)$ is the **old estimate**

**Training Loop** (10,000 episodes):

```python
for episode in range(N_EPISODES):
    s = env.reset()
    for step in range(TMAX):
        # Epsilon-greedy action selection
        if random() < eps:
            a = random_action()  # Exploration
        else:
            a = argmax_a Q(s, a)  # Exploitation
        
        # Environment step
        s_next, r, done = env.step(a)
        
        # Q-learning update
        Q[s][a] += ALPHA * (r + (0.0 if done else GAMMA * max(Q[s_next])) - Q[s][a])
        
        s = s_next
        if done:
            break
    
    # Decay exploration
    eps = max(EPS_END, eps * EPS_DECAY)
```

**Exploration Schedule**:
- **Initial** ($\text{eps} = 1.0$): 100% random actions (explore state space)
- **Final** ($\text{eps} = 0.05$): 95% greedy, 5% random (refine policy)
- **Decay**: $\text{eps} \leftarrow 0.995 \times \text{eps}$ each episode

---

### Learned Policy: Deployment

After training, the policy is deterministic (greedy):

$$\pi(s) = \arg\max_a Q(s, a)$$

**Usage**: For a new listing with segment `seg_id` at week `t` with price delta `Δ`:
1. Discretize: $\Delta \rightarrow \text{delta\_bucket}$
2. Look up: $Q[\text{seg\_id}, t, \text{delta\_bucket}]$
3. Select: $a^* = \arg\max Q[s]$
4. Execute: Adjust price by `ACTION_PCTS[a*]`

---

## Evaluation

### Baseline

1. **Keep Price** (`keep_price`): Never adjust; hold at $p_0(x)$

### Metrics

- **Mean reward per episode**: Average revenue minus holding costs
- **Sale rate**: Fraction of episodes that end in sale (vs. timeout)
- **Survival curves**: Probability of still unsold at week $t$
- **Segment breakdown**: Per-segment average reward
- **Sensitivity analysis**: Robustness to changes in holding cost, discount factor

---

## Configuration & Hyperparameters

All key parameters are defined in **Cell 0** of the notebook:

### Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TMAX` | 12 | Max weeks per episode |
| `ACTION_PCTS` | `[-5%, -3%, 0%, +3%, +5%]` | Discrete price adjustments |
| `HOLDING_COST` | €40 | Weekly cost of unsold listing |

### State Discretization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DELTA_BINS` | 16 | Number of price-deviation bins |
| `Delta range` | [-0.5, 0.5] | Log-price deviation bounds |
| `MIN_MODEL_COUNT` | 200 | Min listings per segment |

### RL Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_EPISODES` | 10,000 | Training episodes |
| `ALPHA` | 0.10 | Q-learning step size |
| `GAMMA` | 0.98 | Discount factor (future importance) |
| `EPS_START` | 1.0 | Initial exploration rate |
| `EPS_END` | 0.05 | Final exploration rate |
| `EPS_DECAY` | 0.995 | Exploration decay per episode |

### Models

**LightGBM Price Model (to estimate fair price)** (`p0(x)`):
- Estimators: 500
- Learning rate: 0.05
- Num leaves: 64
- Subsample: 0.8

---

## Notebook Structure

The main notebook is organized into 10 major sections:

| Section | Title | Purpose |
|---------|-------|---------|
| 0 | Config + Imports | Hyperparameters and dependencies |
| 1 | Load + Clean Data | Parse and preprocess `autos.csv` |
| 2 | Safe Grouping Strategy | Create homogeneous car segments |
| 3 | Baseline Price Model | Train $p_0(x)$ LightGBM regressor |
| 1.5 | Build Ad-ID | Identify repeated listings |
| 4 | Empirical Transitions | Extract $P(\text{sale} \mid \text{segment}, \text{week}, \text{action})$ from real data |
| 5 | RL Environment | Implement `CarPricingEnv` dynamics |
| 6 | Train RL Agent | Run Q-learning for 10k episodes |
| 7 | Baselines + Evaluation | Compare agent vs. baselines |
| 7.5 | Offline Evaluation | Compare to real seller behavior |
| 8–9 | Extended Diagnostics | Price model residuals, OOD detection, Q-value heatmaps, convergence analysis |

---

## Key Technical Insights

### Why Model-Based RL with Empirical Transitions?

Traditional approaches have limitations:
- **Model-free RL**: Requires expensive online exploration on real listings
- **Synthetic transitions**: ML-predicted sale probabilities can be unrealistic or biased

**Empirical transition approach**: Extract $P(\text{sale} \mid \text{segment}, \text{week}, \text{action})$ directly from real price trajectories, then simulate episodes offline. This ensures:
- Realistic sale dynamics (grounded in actual seller behavior)
- Offline training (no live experimentation needed)
- Interpretable transitions (aggregated from real data)
- Fast, stable learning (concrete probability estimates)

### Why Tabular Q-Learning?

- Small state space (~200 segments × 12 weeks × 16 delta bins ≈ 40k states)
- Interpretable Q-values
- Fast convergence (10k episodes)
- Easy policy extraction (greedy argmax)

For larger state spaces or continuous actions, deep RL (DQN, PPO) would be more appropriate.
 (Critical!)

---

## File Structure

```
projeto_tx_codex/
├── README.md                          # This file
├── config.yaml                        # (Optional) External config
├── data/
│   └── autos.csv                      # German used-car dataset
├── rl_pricing_model_based_id.ipynb    # Main notebook (recommended)
├── rl_pricing_model_based_id_DP.ipynb # Variant (full-data transitions)
├── rl_pricing_model_based.ipynb       # Older version
├── notebooks/
│   └── rl_pipeline_simple.ipynb       # Simplified pipeline
├── data_utils.py                      # Data loading/processing helpers
├── env.py                             # CarPricingEnv implementation
├── evaluate.py                        # Evaluation utilities
├── hazard_utils.py                    # Hazard model helpers
├── plan_policy.py                     # Policy deployment utilities
├── train_hazard_model.py              # Standalone hazard model script
├── train_price_model.py               # Standalone price model script
└── tests/

```

---

## Troubleshooting

### Issue: "ImportError: No module named lightgbm"
**Solution**: Install dependencies:
```bash
pip install lightgbm scikit-learn
```

### Issue: "FileNotFoundError: data/autos.csv"
**Solution**: Ensure the dataset exists at the correct path. Place the German used-car CSV in `data/autos.csv`.

### Issue: Q-values not converging (remain near zero)
**Solution**: 
- Check hazard model calibration (should output probabilities ≈ 0.05–0.30)
- Verify empirical transitions exist for your segment
- Increase `N_EPISODES` or decrease `EPS_DECAY` for more exploration

### Issue: Segment counts too small
**Solution**: Increase `MIN_MODEL_COUNT` to include more rare models, or decrease it if you want finer-grained segmentation.
