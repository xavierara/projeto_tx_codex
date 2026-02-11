# RL Pricing for Used-Car Listings

This repository implements a **model-based reinforcement learning pipeline** for dynamic pricing of used-car listings. The system extracts empirical transition probabilities from real price trajectories and trains a tabular Q-learning agent in a simulated environment to optimize pricing decisions, maximizing expected revenue while accounting for holding costs.

**Main Notebook**: [pricing_rl.ipynb](pricing_rl.ipynb)

## Overview

The pipeline operates in three phases:

1. **Offline Learning Phase**: Build price prediction model from historical data, extract empirical transitions from real price trajectories, then train an RL agent in a simulated environment
2. **Deployment Phase**: Use the learned Q-table to recommend prices for new listings
3. **Online Learning (Optional)**: Continuously refine the policy with live feedback from actual sales and market dynamics

---

## Pipeline Walkthrough

### 1. Data Loading & Cleaning

**Purpose**: Prepare raw `autos.csv` for downstream modeling

**Steps**:
- Load German used-car listings dataset
- Parse datetime columns: `dateCreated`, `dateCrawled`, `lastSeen`
- Drop records with missing dates
- Calculate listing duration: `duration_days = (lastSeen - dateCreated)`
- Create `ad_id`: composite key (`brand|model|year|km|gearbox|color|seller`) to identify repeated listings of the same car
- Translate German categorical values to English (e.g., `Benzin` → `Petrol`)
- Filter extreme outliers (prices > 99th percentile, mileage > 99th percentile, power PS, year range)
- Create derived features: `age_years`, `log_km`, `log_power`, `month_created`

**Output**: Clean DataFrame with temporal features, standardized labels, and ad-tracking alongside price trajectory identification

---

### 2. Segmentation: Composite Grouping

**Purpose**: Partition cars into homogeneous segments for modeling

**Strategy**:
```
Segment = (Brand, Model_slim, Vehicle Type, Year Bin)
```

**Details**:
- Extract brand-model pairs; keep only those with ≥200 listings (`MIN_MODEL_COUNT`)
- For rare brand-model pairs, collapse model to `"other"`
- Vehicle type: `Sedan`, `SUV`, `Estate`, `Coupe`, `Convertible`, `Van`, `City Car`, `Off-road`, `Pickup`
- Year bins: 3-year ranges (e.g., `2021–2023`, `2018–2020`)
- Assign each listing a `seg` (string composite key)
- **Fallback**: If segment count > 2000, use lighter grouping: `(Brand, Vehicle Type, Year Bin)`

**Rationale**: Ensures sufficient data per segment for reliable empirical transition estimates while avoiding over-segmentation

---

### 3. Baseline Price Model: $p_0(x)$

**Purpose**: Predict fair-market price for a car given its features

**Data Split**: 80% train / 20% validation (temporal split on `dateCreated`)

**Model**: LightGBM regressor on log-transformed prices

**Features**:
- Numerical: `age_years`, `log_km`, `log_power`, `month_created`
- Categorical: `brand`, `vehicleType`, `gearbox`, `fuelType`, `notRepairedDamage`, `seller`, `model_slim` (if < 300 unique values)

**Training**:
```python
# Fit on log-price target
y = log(price)

lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
).fit(X_train, y_train)
```

**Output**: Function $p_0(x)$ that predicts baseline price in €; validation RMSLE reported

**Usage**: Represents the "fair" price; deviations from $p_0$ are encoded as state delta $\Delta = \log(\text{price} / p_0(x))$

### 4. Empirical Transition Table

**Purpose**: Learn empirical sale probabilities $P(\text{sale} \mid \text{segment}, \text{week}, \text{action})$ from real data

**Data Source**: **Training set only** (80% split, no data leakage into evaluation)

**Process**:

1. **Extract trajectories**: For each ad-id with ≥2 observations, collect price sequence across crawls
   ```
   Trajectory: observation₁ → observation₂ → ... → last observation (assumed sold/delisted)
   ```

2. **Compute price changes & actions**: For each transition within a trajectory
   ```
   price_change_pct = (price_t+1 - price_t) / price_t
   → Bin into action categories: "drop_5%", "drop_3%", "hold", "raise_3%", "raise_5%"
   ```

3. **Aggregate events**:
   ```
   For each (segment, week_position, action_bin):
       P(sale | seg, t, a) = [# times followed by next observation] / [# total times seen]
   ```

4. **Result**: DataFrame with columns:
   - `seg`: segment identifier (string)
   - `t_week`: week position (1–12)
   - `action_bin`: action category (one of 5 price adjustment buckets)
   - `p_sale`: empirical sale probability (0–1)
   - `n_total`: sample count per (seg, t, a) triplet
   - Supporting stats: `avg_price_change`, `std_price_change`

**Key Property**: Clean train/validation split ensures unbiased evaluation; real seller behavior grounds the transition model

---



---

## Reinforcement Learning Framework

The RL agent learns a **policy** $\pi(s) \rightarrow a$ that maps states to pricing actions to maximize cumulative discounted reward.

### State Space: $s = (\text{seg\_id}, t, \Delta)$

| Component | Description | Values |
|-----------|-------------|--------|
| `seg_id` | Segment ID (hash of segment string) | 0 to ~200+ |
| `t` | Weeks in listing | 1–12 |
| `delta_bucket` | Price deviation bin | 16 bins spanning $[-0.5, 0.5]$ |

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
new_price = p0(x) * exp(delta + log(1 + action_pct))
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

### 5. Implement RL Environment

**Class**: `CarPricingEnv`

**Dynamics** (each step):

1. **Current state**: $(x, t, \Delta)$ where:
   - $x$ = car features (segment, p0-value)
   - $t$ = weeks since listing
   - $\Delta$ = log-price deviation from p0

2. **Agent chooses action**: $a \in \{0, 1, 2, 3, 4\}$ corresponding to `ACTION_PCTS = [-5%, -3%, 0%, +3%, +5%]`

3. **Update price**:
   ```
   Δ_new = Δ + log(1 + ACTION_PCTS[a])
   Δ_new = clip(Δ_new, δmin_seg, δmax_seg)  # Segment-specific bounds
   price_new = p0 * exp(Δ_new)
   ```

4. **Sample sale event** (from empirical transitions):
   ```
   p_sale = P(sale | segment, t, action)  [from real data table]
   sale ← Bernoulli(p_sale)
   ```

5. **Transition**:
   - **If sale**: reward $r = \text{price\_new}$, done = True
   - **Else**: reward $r = -\text{HOLDING\_COST}$ (€40/week), $t \to t+1$, continue

**Termination**: Episode ends when sale occurs or $t > \text{TMAX} = 12$ weeks

**Key Property**: Sale probabilities are grounded in **observed seller behavior** from empirical transitions, ensuring realistic training dynamics without synthetic ML predictions

---

### 6. Q-Learning: Policy Training

**Algorithm**: Tabular Q-learning with ε-greedy exploration

**Q-Table**: Maps states to action values (defaultdict initialized to zeros)
```python
Q[(seg_id, t, delta_bucket)] = [Q_0, Q_1, Q_2, Q_3, Q_4]  # 5 action values per state
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
    total_reward = 0.0
    for step in range(TMAX):
        # Epsilon-greedy action selection
        if random() < eps:
            a = random_action()  # Exploration
        else:
            a = argmax_a Q(s, a)  # Exploitation
        
        # Environment step
        s_next, r, done, info = env.step(a)
        
        # Q-learning update
        Q[s][a] += ALPHA * (r + (0.0 if done else GAMMA * np.max(Q[s_next])) - Q[s][a])
        
        total_reward += r
        s = s_next
        if done:
            break
    
    # Decay exploration
    eps = max(EPS_END, eps * EPS_DECAY)
    
    # Track episode return
    ep_returns.append(total_reward)
```

**Exploration Schedule**:
- **Initial** ($\text{eps} = 1.0$): 100% random actions (explore state space)
- **Final** ($\text{eps} = 0.05$): 95% greedy, 5% random (refine policy)
- **Decay**: $\text{eps} \leftarrow 0.995 \times \text{eps}$ each episode

**Output**: Trained Q-table with learned values for visited state-action pairs; rolling average of episode returns monitored

---

---

### 7. Baselines + Evaluation

**Baseline Policies**:

1. **Keep Price** (`baseline_keep_price`): Always select "hold price" action (index 2); never adjust price
2. **RL Q-Learning** (`rl_policy`): Greedy action selection from learned Q-table

**Evaluation Metrics** (over $N_{\text{eval}} = 1000$ episodes):

- **Mean episode return**: Average cumulative reward (revenue - holding costs)
- **Sale rate**: Fraction of episodes that end in sale (non-timeout)
- **Average weeks to sale**: Mean episode length for sold listings
- **Average price ratio** (conditional on sale): Mean final price / p0 for episodes that sold
- **Survival curves**: Probability of remaining unsold at each week $t = 1, \ldots, 12$

**Learned Policy: Deployment**

After training, the policy is deterministic (greedy):

$$\pi^*(s) = \arg\max_a Q(s, a)$$

**Usage**: For a new listing with segment `seg` at week `t` with price delta $\Delta$:
1. Discretize: $\Delta \rightarrow \text{delta\_bucket}$
2. Map segment: `seg` → `seg_id`
3. Look up: $Q[\text{seg\_id}, t, \text{delta\_bucket}]$
4. Select: $a^* = \arg\max Q[s]$
5. Execute: Adjust price by `ACTION_PCTS[a*]`

---

### 8. Online Learning (Interactive Simulation)

**Concept**: Simulate a scenario where the agent receives live feedback from successful sales during list adjustment and updates its policy online (continuously learning).

**Setup**:
- Start with the pre-trained Q-table from section 6
- Simulate $N$ interactive episodes
- After each sale, log the actual transaction price and conditions
- Apply Q-learning updates to reflect the observed outcome
- Decay exploration rate as episodes progress

**Output**:
- Convergence curves showing online refinement
- Action distribution during adaptive learning
- Comparison of online-learned policy vs. offline-trained baseline

---

### 9. Diagnostics and Analysis

Comprehensive diagnostics to validate model quality, RL convergence, and policy robustness:

**9.1 — Model Quality Diagnostics**
- Price-model residual analysis: histogram, QQ-plot, predicted vs. actual scatter
- Residual statistics: mean, std, skewness, kurtosis
- Identifies potential model misspecification or outlier sensitivity

**9.2 — Policy Robustness & Sensitivity**
- Sweep holding cost: $\{5, 10, 20, 50, 100\}$ €/week
- Compare average returns: RL policy vs. keep-price baseline
- Identifies which cost regime favors discounting vs. patience

**9.3 — Segment-Level Performance**
- Per-segment average returns for both policies
- Identifies segments where RL significantly outperforms or underperforms
- Reveals whether discounting is uniformly beneficial or segment-dependent

**9.4 — Price Action Distribution**
- Overall action frequency (%) for RL policy vs. baselines
- RL action mix by week: shows how strategy evolves over list lifetime
- Validates that policy exhibits sensible progression (e.g., higher discounts as time progresses)

**9.5 — Hazard Reliability Diagram**
- Bin predicted sale probabilities into deciles
- Compare predicted vs. observed event rates in simulation
- Validates that empirical transitions are well-calibrated

**9.6 — Q-Value Heatmaps & Greedy Action Maps**
- Value function $V^*(t, \Delta)$ for most-visited segment
- Greedy action $\pi^*(t, \Delta)$ showing pricing strategy over state space
- Q-table coverage: % of reachable states explored during training

**9.7 — Out-of-Distribution (OOD) Detection**
- Track which states during evaluation were never visited during training
- OOD rate by week and delta bucket
- Flags potential model extrapolation risks

**9.8 — Q-Learning Convergence & Stability**
- Rolling mean ± 1 std of episode returns
- Distribution of returns: first 20% episodes vs. last 20% episodes
- Epsilon decay schedule
- Q-value distribution: shows range and spread of learned estimates
- Convergence stats: improvement from early to late training

---

### 10. Exploratory Examples

Detailed walkthroughs and examples of pipeline outputs for interpretability:

**10.1 — Segment Composition Examples**
- Top segments by listing count
- Characteristics: brand, model, vehicle type, year range, price stats
- Distribution across gearbox/fuel types
- Validates that segments capture meaningful car categories

**10.2 — Price Prediction Examples**
- Sample 10 random cars with actual vs. predicted prices
- Prediction statistics: MAE, MEDIAN AE, RMSLE, delta distribution
- Per-segment prediction accuracy breakdown
- Identifies which segments are easier/harder to price

**10.3 — Agent Policy Walkthrough**
- Simulate same demo cars through both policies ("keep price" vs. RL)
- Show week-by-week breakdown: action taken, event probability, outcome
- Compare cumulative returns and final price ratios
- Demonstrates policy differences and RL advantage/disadvantage by scenario

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
├── README.md                          # Project overview
├── TECHNICAL.md                       # This file (technical documentation)
├── DESIGN_DECISIONS.md                # Design rationale and decisions
├── pricing_rl.ipynb                   # Main notebook (all-in-one pipeline)
└── data/
    └── autos.csv                      # German used-car dataset

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
- Check empirical transitions: verify that sale probabilities are reasonable (typically 0.05–0.30)
- Ensure the training set has sufficient price trajectories per segment
- Verify that `HOLDING_COST` is in a sensible range relative to typical car prices
- Increase `N_EPISODES` or decrease `EPS_DECAY` for more exploration
- Check that price changes are being correctly binned into the 5 action categories

### Issue: Segment counts too small
**Solution**: Increase `MIN_MODEL_COUNT` to include more rare models, or decrease it if you want finer-grained segmentation.
