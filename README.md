# tetrio-win-prediction
- **Task:** Predict whether a player wins a replay (`won`) using gameplay statistics and player rating indicators.  
- **Dataset:** Tetr.io Top Players Replays  
- **Source:** https://www.kaggle.com/datasets/n3koasakura/tetr-io-top-players-replays

---

## 1. Problem Formulation

### 1.1 Supervised Learning Task
**Predict `won` based on gameplay-derived features** such as attack efficiency, combo/back-to-back behavior, T-spin usage, pieces per second, and Glicko-2 rating indicators.

### 1.2 Why this task is interesting
Competitive Tetris is a fast-paced strategy game where outcomes depend on measurable in-game decisions. We aim to identify which replay-level patterns most strongly correlate with winning among high-ranked players.

### 1.3 Target and Unit of Analysis
- **Target variable:** `won` (0 = loss, 1 = win)
- **Raw unit:** piece placement (row-level)
- **Modeling unit:** replay/game (`game_id`) after feature aggregation

---

## 2. Dataset Description

### 2.1 Raw Dataset Summary
- Number of rows (raw): [fill in]
- Number of columns: 21
- Each row corresponds to one piece placement event.

### 2.2 Relevant Columns Used
- IDs/time: `game_id`, `subframe`
- Outcome: `won`
- Action/performance: `cleared`, `garbage_cleared`, `attack`, `t_spin`, `btb`, `combo`
- Pressure/context: `immediate_garbage`, `incoming_garbage`
- Skill indicators: `rating`, `glicko`, `glicko_rd`
- Optional piece composition: `placed`

---

## 3. Data Preparation and Cleaning

### 3.1 Initial Data Integrity Checks
- Confirm shape and column names
- Check data types
- Count missing values
- Check exact duplicate rows
- Check duplicate (`game_id`, `subframe`) rows

### 3.2 Validation Rules Applied
Rows are flagged invalid if any of the following occur:
- `won` not in {0,1}
- `subframe < 0`
- Any of (`cleared`, `garbage_cleared`, `attack`, `btb`, `combo`, `immediate_garbage`, `incoming_garbage`) < 0
- `t_spin` not in {`N`, `M`, `F`}
- `rating`, `glicko`, `glicko_rd` missing or negative

### 3.3 Cleaning Decisions
- Exact duplicates: [removed/kept, specify count]
- Invalid rows: [removed count]
- Missing values: [strategy used]
- Type conversions: [list conversions]

### 3.4 Game-Level Consistency Checks
For each `game_id`, verify:
- `won` is consistent
- `rating`, `glicko`, `glicko_rd` are consistent (or define aggregation rule)

### 3.5 Cleaning Summary Table
Include a small table:
- Raw rows
- Rows after duplicate removal
- Rows after validity filtering
- Final usable rows
- Number of unique games

---

## 4. Feature Engineering (Replay-Level)

### 4.1 Aggregation from Placement-Level to Game-Level
Create one row per `game_id`.

### 4.2 Engineered Features
Examples (adjust as implemented):
- **Volume/Duration**
  - `n_placements`
  - `max_subframe`
  - `duration_sec = max_subframe / 600`
- **Speed**
  - `pps = n_placements / duration_sec`
- **Attack Efficiency**
  - `total_attack`
  - `attack_per_piece = total_attack / n_placements`
  - `attack_per_sec = total_attack / duration_sec`
- **Line/Clear Behavior**
  - `total_cleared`
  - `clear_rate = total_cleared / n_placements`
  - `total_garbage_cleared`
- **T-spin Behavior**
  - `tspin_count` (`M` + `F`)
  - `tspin_rate = tspin_count / n_placements`
- **Chain Behavior**
  - `mean_combo`, `max_combo`
  - `mean_btb`, `max_btb`
  - `%moves_combo_gt0`, `%moves_btb_gt0`
- **Incoming Pressure**
  - `mean_incoming_garbage`, `max_incoming_garbage`
  - `mean_immediate_garbage`
- **Skill Indicators**
  - `rating`, `glicko`, `glicko_rd`

### 4.3 Final Modeling Table
- Rows: one per `game_id`
- Target: `won`
- Feature count: [fill in]

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Target Distribution
- Class balance of `won`

### 5.2 Feature Distributions
- Histograms/boxplots for key features (`pps`, `attack_per_piece`, `tspin_rate`, etc.)

### 5.3 Relationship with Outcome
- Compare feature distributions by `won`
- Correlation heatmap (numeric features)

### 5.4 Notes from EDA
- Key trends observed
- Potential feature transformations
- Outlier handling decisions

---

## 6. Data Splitting and Preprocessing Pipeline

### 6.1 Train/Validation/Test Split
- Stratified split by `won` at **game-level**
- Example: 70/15/15 or 70/30 with CV

### 6.2 Preprocessing
- Scale numeric features for models that require scaling (LogReg/SVM/NN)
- Keep same split for all models for fair comparison

### 6.3 Leakage Control
- `game_id` excluded from model inputs
- Compare two setups:
  1. **With** rating features (`rating`, `glicko`, `glicko_rd`)
  2. **Without** rating features  
This tests dependence on player skill priors.

---

## 7. Models

### 7.1 Classical Model 1: Logistic Regression
- Why chosen
- Hyperparameters tried
- Validation results

### 7.2 Classical Model 2: [Random Forest / XGBoost / SVM]
- Why chosen
- Hyperparameters tried
- Validation results

### 7.3 Neural Network Model: MLP
- Architecture (layers, activations, dropout)
- Training settings (optimizer, LR, epochs, batch size)
- Validation results

---

## 8. Error Analysis and Tuning

### 8.1 Misclassification Analysis
- Inspect false positives and false negatives
- Feature patterns in errors

### 8.2 Tuning Actions
- Hyperparameter tuning decisions
- Feature revisions made after analysis

### 8.3 Post-tuning Performance
- Before vs after tuning comparison

---

## 9. Model Evaluation and Comparison

### 9.1 Metrics
Use classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (if applicable)

### 9.2 Results Table
Include all models under:
- Setup A: with rating features
- Setup B: without rating features

### 9.3 Best Model Selection
- Identify best model
- Explain why it likely performed best

---

## 10. Interpretation and Findings

### 10.1 Most Important Predictors
- Coefficients (LogReg) or feature importances (Tree model)
- Optional SHAP/permutation importance

### 10.2 Practical Interpretation
- What gameplay traits are associated with winning?
- Are ratings overshadowing gameplay signals?

### 10.3 Limitations
- Time window bias (dataset only Jan 22–30, 2024)
- Replay-level confounders
- Generalization limitations

---

## 11. Conclusion

- Restate objective
- Summarize best-performing approach
- Summarize key gameplay insights
- Brief recommendation for future work

---

## 12. Reproducibility Appendix

- Random seeds used
- Library versions
- Runtime notes
- File structure and how to run notebook end-to-end
