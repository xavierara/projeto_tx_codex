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

### Action Space: $a \in {0, 1, 2, 3, 4}$

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

2. **Agent chooses action**: $a \in {0, 1, 2, 3, 4}$ corresponding to `ACTION_PCTS = [-5%, -3%, 0%, +3%, +5%]`

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
- Sweep holding cost: ${5, 10, 20, 50, 100}$ €/week
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

## Model Inputs and Outputs

This section provides detailed specifications for all models in the pipeline.

### Price Prediction Model (LightGBM)

**Purpose**: Estimate fair-market baseline price $p_0(x)$ for any used car

**Model Type**: Gradient-boosted decision trees (LightGBM regressor)

**Training Data**: 80% temporal split (~280k listings from training period)

**Input Features**:

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `age_years` | Numerical | Vehicle age at listing | 3.5, 8.2, 15.0 |
| `log_km` | Numerical | Log-transformed odometer reading | log(50000) ≈ 10.82 |
| `log_power` | Numerical | Log-transformed engine power (PS) | log(150) ≈ 5.01 |
| `month_created` | Numerical | Month of listing creation (1-12) | 3 (March), 11 (November) |
| `brand` | Categorical | Car manufacturer | `volkswagen`, `bmw`, `audi` |
| `vehicleType` | Categorical | Body style | `Sedan`, `SUV`, `Estate`, `Coupe` |
| `gearbox` | Categorical | Transmission type | `Manual`, `Automatic` |
| `fuelType` | Categorical | Fuel type | `Petrol`, `Diesel`, `Electric`, `Hybrid` |
| `notRepairedDamage` | Categorical | Damage status | `Yes`, `No` |
| `seller` | Categorical | Seller type | `Private`, `Commercial` |
| `model_slim` | Categorical | Simplified model name (conditional) | `golf`, `3er`, `a4`, `other` |

**Notes on Features**:
- `model_slim` is **only included** if total unique values < 300; otherwise omitted to avoid overfitting
- All categorical features are label-encoded by LightGBM internally
- Numerical features are used without additional scaling (tree models are scale-invariant)

**Target Variable**:
- **Raw target**: `price` (in €, ranging from ~€500 to ~€100,000)
- **Transformed target**: `log(price)` to stabilize variance across price ranges

**Output**:
- **Raw prediction**: Log-price value (e.g., `8.52`)
- **Final output**: $p_0(x) = \exp(\text{prediction})$ in € (e.g., `€5,023`)

**Model Hyperparameters**:
```python
LGBMRegressor(
    n_estimators=500,      # Number of boosting rounds
    learning_rate=0.05,    # Shrinkage rate
    num_leaves=64,         # Max leaves per tree
    subsample=0.8,         # Row sampling fraction
    colsample_bytree=0.8,  # Column sampling fraction
    random_state=42        # Reproducibility seed
)
```

**Performance Metrics** (on validation set):
- **RMSLE** (Root Mean Square Log Error): Primary evaluation metric
- **MAE** (Mean Absolute Error): Average price deviation in €
- **Residual statistics**: Mean, std, skewness, kurtosis

**Usage in Pipeline**:
1. Predict baseline price for each listing: `p0 = model.predict(features)`
2. Calculate price deviation: `delta = log(actual_price / p0)`
3. Use delta as state component for RL agent

---

### Q-Learning Policy Model

**Purpose**: Learn optimal pricing actions to maximize expected revenue per listing

**Model Type**: Tabular Q-learning (state-action value function)

**State Space Definition**:

Each state $s$ is a tuple: $(s_{\text{seg}}, t, \Delta_{\text{bucket}})$

| Component | Description | Cardinality | Example |
|-----------|-------------|-------------|---------|
| `seg_id` | Vehicle segment identifier (hash) | ~1,200 | 42 (VW Golf, Sedan, 2018-2020) |
| `t` | Week since listing creation | 12 | 1, 5, 12 |
| `delta_bucket` | Discretized price deviation bin | 16 | Bin 7 (delta ∈ [-0.03, 0.0]) |

**Total state space**: ~1,200 × 12 × 16 ≈ **230,000 possible states**

**State Construction**:
1. **Segment mapping**: `seg = (brand, model_slim, vehicleType, year_bin)` → hash to integer ID
2. **Week counting**: `t = ceil((current_date - dateCreated).days / 7)`, clipped to [1, 12]
3. **Delta binning**: `delta = log(price / p0)` → discretize into 16 equal bins spanning [-0.5, 0.5]

**Action Space**:

Five discrete price adjustment actions:

| Action Index | Adjustment | Description | Effect on Price |
|-------------|-----------|-------------|-----------------|
| 0 | -5% | Aggressive discount | `new_price = price × 0.95` |
| 1 | -3% | Moderate discount | `new_price = price × 0.97` |
| 2 | 0% | Hold price | `new_price = price` |
| 3 | +3% | Small increase | `new_price = price × 1.03` |
| 4 | +5% | Aggressive increase | `new_price = price × 1.05` |

**Q-Table Structure**:
```python
Q: Dict[(seg_id, t, delta_bucket)] → Array[5]
# Maps each state to 5 Q-values (one per action)
# Example: Q[(42, 3, 7)] = [2450.0, 2680.0, 2590.0, 2310.0, 2180.0]
#          → Action 1 (−3%) has highest value in this state
```

**Output**:
- **During training**: Q-value updates via Bellman equation
- **During deployment**: Greedy policy $\pi^*(s) = \arg\max_a Q(s, a)$ returns optimal action index

**Training Algorithm** (Q-learning):

**Input**: Environment with empirical transitions, initial Q-table (zeros)

**Process** (10,000 episodes):
1. Reset environment → initial state $s_0$
2. For each timestep until episode ends:
   - Select action: $a = \begin{cases} \text{random} & \text{with probability } \epsilon \\ \arg\max_a Q(s,a) & \text{otherwise} \end{cases}$
   - Execute action → observe reward $r$, next state $s'$, done flag
   - Update Q-value: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
   - Decay exploration: $\epsilon \leftarrow \max(\epsilon_{\text{end}}, \epsilon \times \epsilon_{\text{decay}})$
3. Log episode return for convergence monitoring

**Output**: Trained Q-table with values for ~10k visited states

**Deployment Usage**:
```python
# For a listing with features x at week t with current price p:
seg_id = get_segment_id(x)
delta = log(p / p0(x))
delta_bucket = discretize_delta(delta)
state = (seg_id, t, delta_bucket)

# Get optimal action
action = argmax(Q[state])  # Greedy policy

# Execute price adjustment
new_price = p * (1 + ACTION_PCTS[action])
```

---

### Empirical Transition Model

**Purpose**: Provide realistic sale probabilities grounded in observed seller behavior (no ML prediction)

**Model Type**: Lookup table with aggregated empirical frequencies

**Input Data**: Price trajectories from training set only (no data leakage)

**Trajectory Extraction**:
1. Group listings by `ad_id` (composite key: brand|model|year|km|gearbox|color|seller)
2. For ads with ≥2 observations, reconstruct price sequence: $p_1 \to p_2 \to \ldots \to p_n$
3. Last observation assumed to be sale/delisting event

**Transition Labeling**:

For each consecutive pair $(p_t, p_{t+1})$ in a trajectory:
1. **Compute action**: `pct_change = (p_{t+1} - p_t) / p_t`
2. **Bin action**: Map to one of 5 categories:
   - `drop_5%`: change < -4%
   - `drop_3%`: -4% ≤ change < -1.5%
   - `hold`: -1.5% ≤ change < 1.5%
   - `raise_3%`: 1.5% ≤ change < 4%
   - `raise_5%`: change ≥ 4%
3. **Track outcome**: Did listing continue (next observation exists) or sell (end of trajectory)?

**Aggregation Process**:

For each triplet $(s, t, a)$ where:
- $s$ = segment (brand, model_slim, vehicleType, year_bin)
- $t$ = week position (1–12)
- $a$ = action bin (one of 5 categories)

Compute:
```python
P(sale | s, t, a) = count(transitions ending in sale) / count(all transitions)
```

**Output Table Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `seg` | String | Segment key | `"volkswagen|golf|Sedan|2018-2020"` |
| `t_week` | Int | Week position | 3 |
| `action_bin` | String | Action category | `"drop_3%"` |
| `p_sale` | Float | Empirical sale probability | 0.18 (18% chance) |
| `n_total` | Int | Sample size | 247 observations |
| `avg_price_change` | Float | Mean % change in bin | -2.8% |
| `std_price_change` | Float | Std dev of % change | 0.4% |

**Fallback Strategy** (for unseen states):

If exact $(s, t, a)$ triplet not observed during training:
1. Try action-averaged: $\bar{P}(s, t) = \frac{1}{5}\sum_a P(s, t, a)$
2. Try segment-averaged: $\bar{P}(s) = \text{mean over all weeks and actions}$
3. Use global average: $\bar{P}_{\text{global}} = 0.15$ (typical sale rate)

**Usage in RL Environment**:
```python
# During episode step:
seg = get_segment(car_features)
week = current_week
action_bin = map_action_to_bin(chosen_action)

# Query empirical transition table
p_sale = transitions.get((seg, week, action_bin), fallback=0.15)

# Sample outcome
sale_occurred = np.random.binomial(1, p_sale)
```

**Data Statistics** (Training Set):
- **Total trajectories**: ~50k ads with multiple observations
- **Avg trajectory length**: 2.3 observations per ad
- **Coverage**: ~80% of (segment, week, action) triplets observed with n ≥ 10
- **Mean sale probability**: 0.15 (15% per week-action combination)

---

## Model Update and Retraining

This section provides guidelines for maintaining model performance as market conditions evolve.

### When to Retrain Models

**Triggering Conditions** (any of the following):

1. **Temporal Drift** (recommended): Retrain every **3–6 months** to capture seasonal patterns and market shifts
2. **Performance Degradation**: Price model RMSLE increases by >10% on recent data
3. **Policy Underperformance**: RL agent's average return drops below baseline by >15% over 1000 evaluation episodes
4. **Data Volume Threshold**: Accumulated >50k new listings since last training
5. **Market Events**: Major economic shifts, policy changes, or dataset expansions

**Monitoring Metrics**:

| Model | Metric | Threshold for Retraining |
|-------|--------|--------------------------|
| Price Model | Validation RMSLE | >10% increase from baseline |
| Price Model | Residual drift | Mean residual > ±5% on recent data |
| Q-Learning | Avg episode return | <85% of baseline policy return |
| Q-Learning | OOD state rate | >30% of evaluation states unseen during training |
| Transitions | Coverage gaps | >20% of validation states lack empirical data |

---

### Retraining Procedure: Price Prediction Model

**Objective**: Update $p_0(x)$ to reflect current market prices

**Data Requirements**:
- Minimum **100k listings** spanning ≥6 months
- Recent data (last 12 months preferred)
- Balanced representation across vehicle types, brands, price ranges

**Steps**:

1. **Prepare Fresh Dataset**:
   ```python
   # Load new data
   df_new = pd.read_csv("data/autos_recent.csv")
   
   # Apply same preprocessing as original pipeline:
   # - Parse dates, calculate duration_days
   # - Create ad_id composite keys
   # - Translate German → English
   # - Filter outliers (price, mileage, power, year)
   # - Engineer features: age_years, log_km, log_power, month_created
   ```

2. **Merge with Historical Data** (optional, for stability):
   ```python
   # Combine last 12 months of historical + new data
   df_train = pd.concat([df_historical[-12mo:], df_new]).drop_duplicates()
   ```

3. **Perform Temporal Split** (80/20):
   ```python
   cutoff_date = df_train['dateCreated'].quantile(0.80)
   train = df_train[df_train['dateCreated'] <= cutoff_date]
   val = df_train[df_train['dateCreated'] > cutoff_date]
   ```

4. **Retrain LightGBM**:
   ```python
   # Use same hyperparameters as original training
   model_new = lgb.LGBMRegressor(
       n_estimators=500, learning_rate=0.05,
       num_leaves=64, subsample=0.8,
       colsample_bytree=0.8, random_state=42
   )
   
   # Fit on log-price
   model_new.fit(X_train, np.log(y_train))
   ```

5. **Validate Performance**:
   ```python
   # Compute RMSLE on validation set
   y_pred_val = np.exp(model_new.predict(X_val))
   rmsle_new = np.sqrt(mean_squared_error(np.log(y_val), np.log(y_pred_val)))
   
   # Compare to previous model
   if rmsle_new < rmsle_old * 1.1:  # Allow 10% tolerance
       print("✓ New model approved")
   else:
       print("✗ Performance regression detected")
   ```

6. **Save Versioned Model**:
   ```python
   # Pickle with version tag
   import pickle
   from datetime import datetime
   version = datetime.now().strftime("%Y%m%d")
   pickle.dump(model_new, open(f"models/price_model_v{version}.pkl", "wb"))
   ```

**Backward Compatibility**:
- Ensure feature names/types match original pipeline
- Log version metadata: training date, data range, RMSLE, model parameters

---

### Retraining Procedure: Empirical Transition Model

**Objective**: Update sale probabilities $P(\text{sale} \mid s, t, a)$ using recent seller behavior

**Prerequisites**: Updated price model $p_0(x)$ already trained

**Steps**:

1. **Extract Trajectories from New Data**:
   ```python
   # Group by ad_id, sort by dateCrawled
   trajectories = df_train.groupby('ad_id').apply(
       lambda g: g.sort_values('dateCrawled')[['price', 'seg', 'dateCreated', 'lastSeen']]
       if len(g) >= 2 else None
   ).dropna()
   ```

2. **Compute Actions and Outcomes**:
   ```python
   for ad_id, traj in trajectories.items():
       for i in range(len(traj) - 1):
           price_t = traj.iloc[i]['price']
           price_t1 = traj.iloc[i+1]['price']
           pct_change = (price_t1 - price_t) / price_t
           
           # Bin action
           action_bin = bin_price_change(pct_change)  # One of 5 categories
           
           # Label outcome
           is_last_obs = (i == len(traj) - 2)
           outcome = 'sale' if is_last_obs else 'continue'
           
           # Record transition
           seg = traj.iloc[i]['seg']
           week = compute_week(traj.iloc[i]['dateCreated'], traj.iloc[i]['dateCrawled'])
           transitions_log.append((seg, week, action_bin, outcome))
   ```

3. **Aggregate Empirical Probabilities**:
   ```python
   # Group by (seg, week, action), compute P(sale)
   transition_table = transitions_log.groupby(['seg', 't_week', 'action_bin']).apply(
       lambda g: pd.Series({
           'p_sale': (g['outcome'] == 'sale').mean(),
           'n_total': len(g),
           'avg_price_change': g['pct_change'].mean(),
           'std_price_change': g['pct_change'].std()
       })
   ).reset_index()
   ```

4. **Validate Coverage**:
   ```python
   # Check that ≥80% of validation states have n_total ≥ 10
   coverage = (transition_table['n_total'] >= 10).mean()
   print(f"Coverage: {coverage:.1%}")  # Should be ≥ 0.80
   ```

5. **Save Versioned Transition Table**:
   ```python
   transition_table.to_csv(f"models/transitions_v{version}.csv", index=False)
   ```

**Important**: Re-extract transitions from **training split only** to avoid data leakage.

---

### Retraining Procedure: Q-Learning Policy

**Objective**: Update Q-table using new empirical transitions and market conditions

**Prerequisites**: 
- Updated price model $p_0(x)$
- Updated empirical transition table

**Steps**:

1. **Initialize RL Environment with New Models**:
   ```python
   env = CarPricingEnv(
       segments=segments_new,
       price_model=model_new,  # Updated p0(x)
       transitions=transition_table_new,  # Updated P(sale | s,t,a)
       holding_cost=HOLDING_COST  # May adjust if market conditions changed
   )
   ```

2. **Warm-Start Q-Table (Optional)**:
   ```python
   # Transfer learned values for common states
   Q_new = defaultdict(lambda: np.zeros(5))
   
   for state, values in Q_old.items():
       if state_exists_in_new_segments(state):
           Q_new[state] = values  # Initialize with old policy
   ```

3. **Retrain with Q-Learning** (10k episodes):
   ```python
   for ep in range(N_EPISODES):
       s = env.reset()
       total_reward = 0
       
       for t in range(TMAX):
           # Epsilon-greedy action selection
           if np.random.rand() < eps:
               a = np.random.randint(5)
           else:
               a = np.argmax(Q_new[s])
           
           # Step environment
           s_next, r, done, info = env.step(a)
           
           # Q-learning update
           td_target = r + (0 if done else GAMMA * np.max(Q_new[s_next]))
           Q_new[s][a] += ALPHA * (td_target - Q_new[s][a])
           
           total_reward += r
           s = s_next
           if done:
               break
       
       eps = max(EPS_END, eps * EPS_DECAY)
   ```

4. **Evaluate New Policy**:
   ```python
   # Run 1000 evaluation episodes on validation set
   avg_return_new = evaluate_policy(Q_new, env, n_episodes=1000)
   avg_return_baseline = evaluate_policy(baseline_policy, env, n_episodes=1000)
   
   # Compare performance
   improvement = (avg_return_new - avg_return_baseline) / abs(avg_return_baseline)
   print(f"RL vs Baseline: {improvement:+.1%} improvement")
   ```

5. **A/B Testing (Recommended)**:
   ```python
   # Before full deployment, test new policy on 10% of listings
   # Compare actual revenues after 4 weeks
   if new_policy_revenue >= old_policy_revenue * 0.95:
       deploy_policy(Q_new, version)
   else:
       print("✗ A/B test failed, reverting to previous policy")
   ```

6. **Save Versioned Q-Table**:
   ```python
   with open(f"models/qtable_v{version}.pkl", "wb") as f:
       pickle.dump(Q_new, f)
   
   # Save metadata
   metadata = {
       'version': version,
       'n_states': len(Q_new),
       'avg_return': avg_return_new,
       'training_episodes': N_EPISODES,
       'price_model_version': price_model_version,
       'transitions_version': transitions_version
   }
   json.dump(metadata, open(f"models/qtable_v{version}_meta.json", "w"))
   ```

---

### Model Versioning and Deployment Best Practices

**Version Naming Convention**:
```
{model_name}_v{YYYYMMDD}_{optional_tag}.{ext}

Examples:
- price_model_v20260212.pkl
- transitions_v20260212.csv
- qtable_v20260212_highcost.pkl  (if testing different holding cost)
```

**Deployment Checklist**:
- [ ] Validate all three models on held-out test set (most recent 2 weeks of data)
- [ ] Compute baseline metrics: RMSLE, avg return, sale rate
- [ ] Document changes in model performance vs. previous version
- [ ] Test edge cases: rare segments, extreme prices, week 12 states
- [ ] Run A/B test on 10% traffic for 2–4 weeks (production environments)
- [ ] Monitor daily: prediction errors, policy actions, revenue per listing
- [ ] Maintain rollback capability: keep previous 3 versions accessible

**Performance Monitoring Dashboard** (recommended metrics):
- **Price model**: Daily RMSLE, residual distribution, prediction latency
- **RL policy**: Weekly avg return, action distribution, sale rate, time-to-sale
- **Transitions**: Coverage rate, OOD state frequency, empirical vs. actual sale rates

**Automated Retraining Pipeline** (Optional):
```python
# Pseudo-code for scheduled retraining
if (current_date - last_training_date).days >= 90:  # Every 3 months
    df_recent = fetch_data(last_training_date, current_date)
    
    # Train all three models
    price_model = train_price_model(df_recent)
    transitions = extract_transitions(df_recent, price_model)
    qtable = train_rl_policy(price_model, transitions)
    
    # Validate
    if validate_models(price_model, transitions, qtable):
        deploy_models(version=current_date.strftime("%Y%m%d"))
        send_notification("✓ Models retrained successfully")
    else:
        send_alert("✗ Model validation failed")
```

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
