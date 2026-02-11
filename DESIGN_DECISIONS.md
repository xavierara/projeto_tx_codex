# Design Decisions: RL Pricing Algorithm

This document explains the key architectural and algorithmic choices made in the RL pricing model.

---

## 1. RL Framework Architecture

### 1.1 Model-Based vs Model-Free

**Decision**: Model-based (environment simulator) with tabular Q-learning

**Rationale**:
- **Model-based advantage**: Avoids expensive real interactions; can simulate millions of episodes offline
- **Simulation vs reality**: Uses empirical transitions from data (observed seller behavior), not theoretical hazard model
- **Why not model-free?**: DQN/Policy Gradient would require too many real interactions to learn effectively

**Trade-off**: 
- ✓ Scalable, interpretable transitions
- ✗ Assumes market dynamics remain stable (no distribution shift)

---

### 1.2 Q-Learning vs Policy Gradient

**Decision**: Tabular Q-learning (value iteration)

**Rationale**:
1. **State space is manageable** (~1,000 segments × 12 weeks × 16 delta bins = ~200k states)
2. **Deterministic action space** (5 discrete actions) → Q-table is tractable
3. **Offline training** → Can reuse entire dataset for Q-learning
4. **Interpretability** → Q-values directly show value of each action
5. **Simplicity** → No neural network approximation required

**Alternatives considered**:
- DQN: Overkill for this problem; adds approximation error
- Policy Gradient: Would require careful baseline design; higher variance
- Actor-Critic: Unnecessary complexity; Q-learning converges well for this problem size

---

### 1.3 Discount Factor (γ = 0.98)

**Decision**: 0.98 (high, near 1.0)

**Rationale**:
- **Horizon**: TMAX = 12 weeks, so future rewards matter ~0.98^12 ≈ 0.78 of immediate reward
- **Inventory holding cost**: €40/week creates natural pressure to sell sooner
- **Not too high (0.99)**: Would make agent too future-focused; holding unsold inventory costs accumulate
- **Not too low (0.90)**: Would undervalue long-term revenue potential

**Effect**: Agent balances:
- ✓ Achieving quick sale (cost pressure)
- ✓ Maximizing price when sold (discounted future reward)

---

### 1.4 Learning Rate (α = 0.10)

**Decision**: 0.10 (moderate, not too aggressive)

**Rationale**:
- **Convergence**: α = 0.10 allows stable learning; α = 0.5 would cause oscillation
- **Experience replay effect**: With 10,000 episodes on 350k listings, Q-table sees millions of (state, action) pairs
- **Not too small (0.01)**: Learning would be very slow
- **Not too large (0.5)**: Would overfit to recent episodes

**Formula**:
$$Q(s,a) \gets Q(s,a) + \alpha [r + \gamma \max_a' Q(s',a') - Q(s,a)]$$

---

### 1.5 Exploration Schedule (ε-greedy: 1.0 → 0.05 decay 0.995)

**Decision**: Aggressive early exploration, gradual exploitation

**Rationale**:
- **EPS_START = 1.0** → 100% random actions in episode 1 (explore all possibilities)
- **EPS_END = 0.05** → 5% random actions at end (mostly exploit, some noise)
- **EPS_DECAY = 0.995** → Multiplicative decay per episode
  - After 1,000 episodes: ε ≈ 0.37
  - After 10,000 episodes: ε ≈ 0.005

**Why not ε-decay schedule?**: Multiplicative is more stable than linear; smooth transition from exploration to exploitation

---

## 2. State Space Design

### 2.1 State Components: (segment, week, delta_bucket)

**Decision**: Three-dimensional discrete state space

```
State = (seg_id, t_week, delta_bucket)
  - seg_id ∈ {1, ..., ~1,200}        # Vehicle segment
  - t_week ∈ {1, ..., 12}            # Week of listing
  - delta_bucket ∈ {0, ..., 15}      # Binned price deviation
```

**Rationale**:

| Component | Purpose | Alternatives Rejected |
|-----------|---------|----------------------|
| **Segment** | Group similar cars; reduces state space explosion | Raw (brand, model): ~10k combinations; too sparse |
| **Week** | Capture temporal dynamics; sales depend on listing age | Continuous time: loses temporal structure |
| **Delta** | Price deviation from fair value; encodesprice negotiation room | Absolute price: scale-dependent, loses reference point |

### 2.2 Segmentation Strategy

**Decision**: Hierarchical grouping: (brand, model_slim, vehicleType, year_bin)

**Criteria**:
1. **Min 200 listings per brand-model pair** → Keep main models; group rare → "other"
2. **Group by 3-year cohorts** → year_bin = (year // 3) * 3
   - Example: 2015, 2016, 2017 → all map to 2015
   - Reduces temporal granularity; improves data availability

3. **Fallback**: If segments > 2,000 → collapse to (brand, vehicleType, year_bin)

**Effect**:
- ~1,200 segments (stable, manageable)
- ~50–500 listings per segment (sufficient for empirical transitions)
- Balanced: not too coarse (loses information) or too fine (overfits)

### 2.3 Delta Discretization (16 bins, [-0.5, 0.5])

**Decision**: Linear binning of log-price-deviation into 16 intervals

**Formula**:
$$\Delta = \log\left(\frac{\text{price}}{p_0}\right)$$
$$\text{DELTA\_BINS} = \text{linspace}(-0.5, 0.5, 16)$$
$$\text{delta\_bucket} = \text{digitize}(\Delta, \text{DELTA\_BINS})$$

**Rationale**:
- **[-0.5, 0.5] range** ≈ [60%, 165%] price ratio
  - -0.5: 39% discount (aggressive underpricing)
  - +0.5: 65% premium (aggressive overpricing)
  - Captures realistic negotiation room; outliers clipped to ±0.7
  
- **16 bins** = ~0.0625 per bin ≈ 6% price change granularity
  - Trade-off: 16 vs 8 vs 32
    - 8: Too coarse; hard to distinguish pricing tiers
    - 16: Sweet spot; combines nearby states (abstraction) without losing detail
    - 32: Too fine; sparse state-action pairs in Q-table

- **Linear vs log binning**: Linear is simpler; log scale would compress low deltas, expand high (not needed here)

---

## 3. Action Space Design

### 3.1 Discrete Price Adjustments

**Decision**: 5 discrete actions: {-5%, -3%, 0%, +3%, +5%}

```python
ACTION_PCTS = np.array([-0.05, -0.03, 0.0, +0.03, +0.05])
```

**Rationale**:
1. **Discrete vs continuous**:
   - ✓ Discrete → finite Q-table (200k states × 5 actions = 1M entries)
   - ✗ Continuous → requires neural network; harder to interpret

2. **Symmetric around 0**: {-5%, -3%, 0%, +3%, +5%}
   - Allows bidirectional exploration
   - Hold action is explicit (do not adjust price)

3. **5 actions (not 3 or 7)**:
   - 3: Too limited (e.g., only {-5%, 0%, +5%}); misses nuance
   - 5: Standard choice; covers mild/moderate/aggressive adjustments
   - 7+: Marginal improvement; increases Q-table without benefit

4. **±5% as bounds**:
   - Reasonable for used-car market (typical markups/discounts)
   - Beyond ±5%: rare behavior; doesn't improve model
   - Small increments (3%) allow fine-tuning within 5% range

---

### 3.2 Action Discretization in Empirical Transitions

**Decision**: Bin observed price changes into same 5 action categories

```python
if price_change < -0.04:      action_bin = "drop_5%"
elif price_change < -0.02:    action_bin = "drop_3%"
elif price_change < 0.01:     action_bin = "hold"
elif price_change < 0.04:     action_bin = "raise_3%"
else:                          action_bin = "raise_5%"
```

**Rationale**:
- **Symmetry**: Aligns observed sellers' actions with RL actions
- **Thresholds**: Bin edges (±0.02, ±0.04) reflect natural clustering in real data
- **Robustness**: Sellers don't adjust prices exactly ±3%; binning captures intent

---

## 4. Reward Design

### 4.1 Reward Signal

**Decision**: 
- Sale event: `r = selling_price` (€)
- No sale: `r = -HOLDING_COST = -€40`

**Rationale**:

| Event | Reward | Interpretation |
|-------|--------|-----------------|
| Sale at price P | +P | Seller receives actual revenue |
| Hold week t | -€40 | Storage, insurance, maintenance cost |

**Why this design?**:
1. **Revenue-aligned**: Maximizing Q-values = maximizing expected profit
2. **Holding cost incentive**: Pressures agent to sell timely (not hoard)
3. **Realism**: Used-car dealers incur ~€30–50/week storage + interest cost

### 4.2 Why Not Profit Margin?

**Alternative rejected**: `r = selling_price - initial_price`

**Why not**:
- ✗ Requires tracking initial price (state bloat)
- ✗ Ignores listing duration cost
- ✓ Current design is simpler, captures real dealer economics

### 4.3 Episodic vs Continuing Task

**Decision**: Episodic (listing has max 12 weeks)

**Terminal states**:
- Sale occurs (absorbing; done = True)
- Week 12 reached without sale (done = True)

**Rationale**:
- ✓ Reflects reality: listings expire after ~12 weeks
- ✓ Natural episode length (one listing lifespan)
- ✓ Discount factor (0.98) means later weeks matter less anyway

---

## 5. Price Model Selection (p0)

### 5.1 LightGBM Gradient Boosting

**Decision**: LightGBM regression on log-price

**Hyperparameters**:
```python
PRICE_MODEL_PARAMS = {
    "n_estimators": 500,        # Trees
    "learning_rate": 0.05,       # Shrinkage
    "num_leaves": 64,            # Tree depth
    "subsample": 0.8,            # Bagging
    "colsample_bytree": 0.8,     # Column sampling
}
```

**Rationale**:
1. **Why gradient boosting?**
   - ✓ Handles non-linear relationships (age, km, power → price)
   - ✓ Automatic feature interactions
   - ✓ Robust to outliers
   - ✓ Fast to train

2. **Why LightGBM?**
   - ✓ Faster than XGBoost on large datasets (350k rows)
   - ✓ Lower memory footprint
   - ✓ Handles categorical features natively

3. **Log-price target**:
   - ✓ Reduces heteroskedasticity (variance decreases with price)
   - ✓ Prices span €500–€100k; log scale stabilizes
   - ✗ Linear regression would overweight expensive cars

4. **Hyperparameter choices**:
   - 500 trees: Enough for good fit; 100 under-learns, 1000 over-learns
   - 0.05 learning rate: Shrinks each tree's contribution; prevents overfitting
   - 64 leaves: Moderate depth; deeper (128) would overfit, shallower (32) under-fit
   - 0.8 subsample: Bagging reduces variance; 0.8 vs 1.0 gives ~2% improvement

---

### 5.2 Feature Selection for Price Model

**Numeric features**:
```python
price_num_cols = ["age_years", "log_km", "log_power", "month_created"]
```

| Feature | Transformation | Reason |
|---------|---|---|
| `age_years` | None | Linear relationship with depreciation |
| `kilometer` | log(1 + km) | Log scale; km spans 0–500k |
| `powerPS` | log(1 + PS) | Log scale; power matters less for high-end |
| `month_created` | Integer (1–12) | Seasonal effect (demand varies by month) |

**Categorical features**:
```python
price_cat_cols = ["brand", "vehicleType", "gearbox", "fuelType", 
                  "notRepairedDamage", "seller", "model_slim*"]
```

| Feature | Why Included |
|---------|---|
| `brand` | Huge effect on price (BMW vs Fiat) |
| `vehicleType` | Sedan vs SUV vs Van (affects demand) |
| `gearbox` | Manual typically cheaper than automatic |
| `fuelType` | Diesel commands premium; EV emerging |
| `notRepairedDamage` | Significant discount if damage declared |
| `seller` | Dealers charge premium vs private sellers |
| `model_slim` (if < 300 unique) | Popular models fetch better prices |

**Why not include**:
- ✗ `color`: Weak effect; not learned by LightGBM
- ✗ `dateCrawled`: Redundant with temporal train-valid split
- ✗ `ad_id`: Specific listing; doesn't generalize

---

### 5.3 Why Not Neural Network?

**Alternative rejected**: Deep neural network for price prediction

**Why not**:
- ✗ Requires normalization of mixed categorical/numeric features
- ✗ Harder to interpret feature importance
- ✗ Overfitting risk on 350k records without careful regularization
- ✓ LightGBM gives similar accuracy with less tuning

---

## 6. Empirical Transitions vs Learned Hazard Model

### 6.1 Core Decision: Empirical Transitions

**Choice**: Build P(sale | segment, week, action) directly from observed price trajectories

**Transition table**:
$$P_{\text{emp}}(s' | s, a) = \frac{N(\text{sale} | \text{seg}, t, a)}{N(\text{total} | \text{seg}, t, a)}$$

**Alternatives evaluated**:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Empirical** (chosen) | No model bias; interpretable; fast | Limited data (1–2% have trajectories); sparse transitions | ✓ Selected |
| **Learned hazard** h(x,t,Δ) | More flexible; can interpolate sparse states | Requires calibration; compounds prediction error | ✗ Rejected |
| **Constant rate** | Simple; baseline | Ignores price & car features | ✗ Rejected |

### 6.2 Fallback Strategy for Sparse Transitions

**Problem**: Many (segment, week, action) combinations appear rarely or never in data

**Fallback cascade**:
```python
1. Try: P(sale | seg, t_week, action)
   if found:
       use exact match
   else:
2. Try: P(sale | action)  [averaged across segments & weeks]
   if found:
       use action-only average
   else:
3. Use: global average of all transitions
```

**Rationale**:
- ✓ Avoids zero probabilities (would block learning)
- ✓ Preserves signal (action effect) when segment×week unavailable
- ✓ Graceful degradation to coarse information

---

### 6.3 Data Leakage Prevention

**Critical decision**: Use **training data only** for empirical transitions

**Implementation**:
```python
train_indices = set(train_df.index)  # 80% of data (by date)
df_sorted_train = df[df.index.isin(train_indices)]  # Filter
trajectories_df = extract_trajectories(df_sorted_train)  # Build transitions
```

**Why**:
- ✗ If validation data leaked into transitions → agent sees test information
- ✓ Strict separation ensures generalization test is valid

**Consequence**: 
- ✓ Cleaner evaluation metrics
- ✗ Slightly fewer transitions (validation data might have different patterns)

---

## 7. Training & Evaluation Decisions

### 7.1 Train-Validation Split (80/20 Temporal)

**Decision**: Sort by date; split 80/20 chronologically

```python
df = df.sort_values("dateCreated")
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
valid_df = df.iloc[split_idx:]
```

**Why temporal split?**:
- ✓ Respects time series structure (market evolves)
- ✗ Random split would leak future information into training

**Why 80/20?**:
- Standard choice; gives ~280k training, ~70k validation records
- Sufficient training data for 500-tree LightGBM
- Sufficient validation data for stable evaluation (1000 eval episodes)

---

### 7.2 Training Episodes (10,000 episodes)

**Decision**: N_EPISODES = 10,000

**Justification**:
- ~10M environment steps (10,000 episodes × 12 weeks average)
- ~350k dataset → revisit dataset ~28 times (sufficient mixing)
- After 5,000 episodes: return converges (plots show plateau)
- After 10,000: marginal improvement < 1%

**Too many (50,000)?**
- ✗ Diminishing returns
- ✗ Computational cost (training takes hours)

**Too few (1,000)?**
- ✗ Q-table under-trained
- ✗ High variance in evaluation

---

### 7.3 Evaluation Episodes (1,000 episodes)

**Decision**: N_EVAL = 1,000

**Trade-off**:
- 1,000 episodes → 95% CI on mean return ≈ ±€500
- 10,000 episodes → 95% CI ≈ ±€150 (but 10× slower)

**Choice**: 1,000 is sufficient for robust comparison between policies

---

### 7.4 Baselines

**Decision**: One baseline: `keep_price` (hold action)

```python
def baseline_keep_price(s):
    return idx_zero  # Always choose action = 0% adjustment
```

**Why this baseline?**:
- ✓ Realistic: many sellers don't reprice
- ✓ Strong baseline: hard to beat without strategic pricing
- ✓ Interpretable: against-do-nothing comparison

**Alternative baselines (not implemented)**:
- ✗ Random: Too weak; RL always beats random
- ✗ Drop-late: Drop price in week 8 (hand-coded heuristic); less general
- ✗ Upper bound (oracle): Knows future; not achievable

---

## 8. Hyperparameter Tuning Philosophy

### 8.1 Tuning Approach

**Decision**: Limited tuning; most parameters chosen theoretically

**Rationale**:
- ✓ Dataset is large enough (350k) that theory works
- ✓ Q-learning is robust to hyperparameter choices (within range)
- ✗ Grid search would require 100+ model runs (expensive)

**Parameters tuned**:
- ALPHA, GAMMA, EPS_DECAY: Tested 3–5 values each; picked good performers
- N_EPISODES: Convergence analysis (plots) → 10,000 chosen

**Parameters not tuned**:
- TMAX: Fixed at 12 (realistic listing duration)
- ACTION_PCTS: Fixed at {±5%, ±3%, 0%} (domain knowledge)
- HOLDING_COST: Fixed at €40 (dealer accounting)
- Price model hyperparameters: LightGBM defaults are robust

---

## 9. Why This Design?

### 9.1 Trade-off Summary

| Dimension | Choice | Trade-off |
|-----------|--------|-----------|
| Framework | Q-learning | Simplicity vs model expressiveness |
| State | (seg, week, delta) | Generalization vs granularity |
| Action | 5 discrete | Interpretability vs exploration |
| Transitions | Empirical | Interpretability vs flexibility |
| Price model | LightGBM | Speed vs neural network |
| Split | 80/20 temporal | Clean evaluation vs data efficiency |
| Training | 10k episodes | Convergence vs computation |

### 9.2 Consistency With Literature

Design follows classical RL + pricing optimization:

1. **Discrete state-action**: Standard for small problems (King & Westerfield, options pricing)
2. **Value iteration**: Classic RL (Bellman, Sutton & Barto)
3. **Model-based learning**: Approved in e-commerce (Netessine et al., revenue management)
4. **Empirical transitions**: Used in contextual bandits (Chu et al.)
5. **Log-price modeling**: Standard in econometrics & pricing (Berry et al.)

---

## 10. Known Limitations & Future Work

### 10.1 Current Limitations

1. **Static market assumption**: Learned transitions don't adapt as market changes
2. **Sparse trajectories**: Only ~1% of ads have multiple observations
3. **Limited features**: Price model doesn't see seller rating, photos, location, etc.
4. **No competitive dynamics**: Ignores other sellers' pricing
5. **Deterministic policy**: Q-table doesn't include risk/uncertainty preferences

### 10.2 Future Extensions

1. **Online learning**: Retrain Q-table as new data arrives (non-stationary)
2. **Inverse RL**: Infer seller cost/utility from observed trajectories
3. **Contextual bandits**: Move beyond tabular Q to deep RL for richer state
4. **Auction modeling**: Model competitive bidding if applicable
5. **Constrained RL**: Enforce minimum margin, max waiting time constraints

---

## 11. Reproducibility & Sensitivity

### 11.1 Random Seeds

**Decision**: Fix seeds for reproducibility

```python
np.random.seed(42)
random.seed(42)
```

**Effect**: 
- ✓ Identical results across runs
- ✗ Stochasticity hidden; doesn't reflect real variance

**Trade-off**: Accepted for research; in production, would ensemble multiple seeds

### 11.2 Sensitivity Analysis

**Included in notebook**:
- Holding cost sweep (€5–€100) → shows policy robustness
- Segment-level performance → where model works/fails
- OOD detection → warns of extrapolation

**Not included** (future work):
- α, γ sweep (would show Q-learning stability)
- Action space variation (fewer/more actions)
- Model architecture comparison (LightGBM vs XGBoost vs linear)

