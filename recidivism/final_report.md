# Two-Year Recidivism: Actuarial Model — Final Report

**Data**: ProPublica COMPAS two-year recidivism file (Broward County, FL, screenings 2013–2014).
**Reproducibility**: `python3 pipeline.py` (seed = 42 throughout). Outputs: `iteration_log.md`, `results.json`, `plots/`, `final_model.joblib`.

---

## TL;DR

The winning model is a **three-variable logistic regression: age, prior count, and priors-per-adult-year**. On the held-out test set it scores **AUC 0.738, Brier 0.207**, versus the two-variable age+priors benchmark (Dressel & Farid) at **AUC 0.739, Brier 0.209**, and the commercial COMPAS decile score at **AUC 0.713, Brier 0.212**.

**The headline finding is a null result**: six hypothesis-driven iterations — juvenile record, prior-offense rate, age splines, charge degree, tuned gradient boosting, isotonic calibration — produced exactly one keepable improvement, worth ~0.003 Brier on validation and essentially zero AUC on test. Everything this dataset can predict about two-year rearrest is captured by age and prior record; no feature engineering or model-class upgrade moves the needle beyond noise. This replicates Dressel & Farid (2018) and Angelino et al.: simple, interpretable rules match both complex ML and the commercial tool.

A second campaign (Rounds 2–2b below, iterations 7–15) pushed past the plateau with nine more hypotheses — jail length-of-stay, charge-type categories, kitchen-sink regularized models, a wider GBM search, calibration, ensembling, and the COMPAS score itself as a feature. It produced real-looking validation gains (best val Brier 0.2016 vs 0.2081) that **evaporated entirely on test** (AUC 0.734 / Brier 0.207 — identical to the three-variable model). The null result survived a determined attempt to break it.

---

## Phase 1 — Data and baselines

- Raw rows: 7,214 → after ProPublica's standard filter (`days_b_screening_arrest` ∈ [−30, 30], `is_recid ≠ −1`, charge degree ≠ "O", valid `score_text`): **6,172 rows**, no missing values in modeling columns.
- Base rate of two-year recidivism: **45.5%**.
- Split: 60/20/20 stratified (train 3,702 / validation 1,235 / test 1,235). The test set was touched exactly once, in Phase 3.

Validation-set baselines:

| Baseline | AUC | Brier |
|---|---|---|
| B1 — base-rate predictor | 0.500 | 0.2480 |
| B2 — logistic(age, priors_count) | 0.7242 | 0.2109 |
| B3 — COMPAS decile (mapped to train decile recidivism rates) | 0.7107 (0.7158 raw-decile AUC) | 0.2150 |

Two variables already beat the commercial tool. Calibration plots: `plots/calibration_B2_val.png`, `plots/calibration_B3_val.png`.

## Phase 2 — Iteration loop (full log in `iteration_log.md`)

Decision rule: keep a change only if validation Brier improves by more than 1 CV standard error (primary), or AUC improves beyond 1 SE with Brier within noise (secondary). Noise estimated by 5-fold CV on the training set.

| # | Change | Val Brier | Val AUC | Decision |
|---|---|---|---|---|
| 1 | + juvenile counts (fel/misd/other) | 0.2105 | 0.7254 | Reverted — ΔBrier −0.0004 < 1 SE |
| 2 | + priors_per_year (prior-offense *rate*) | 0.2081 | 0.7308 | **Kept** — Brier improved beyond 1 SE |
| 3 | age → cubic spline (5 knots) | 0.2088 | 0.7282 | Reverted — slightly worse |
| 4 | + charge_felony | 0.2067 | 0.7344 | Reverted — ΔBrier −0.0014 < 1 SE (0.0023) |
| 5 | LightGBM, grid-tuned by 5-fold CV | 0.2076 | 0.7301 | Reverted — within noise of the logistic |
| 6 | Isotonic calibration of champion | 0.2075 | 0.7310 | Reverted — within noise |

Stopped after 6 iterations: **4 consecutive non-improvements** (the convergence criterion). The plateau arrived fast and across move types — feature engineering, nonlinearity, model class, and recalibration all failed the noise gate.

**Parsimony rule**: five configurations sat within 1 SE of the best validation Brier (0.2067); the simplest — logistic(age, priors_count, priors_per_year) — was selected.

The one keeper supports its hypothesis: prior offense **rate** (priors per year of adulthood) separates high-rate offenders from older offenders whose long careers inflate the raw count. Notably, a tuned LightGBM could not beat this three-variable linear model — the signal in this data is low-dimensional and nearly linear.

## Phase 3 — Test set (touched once)

Winning model retrained on train+validation (4,937 rows), then evaluated once on the 1,235-row test set:

| Model | Test AUC | Test Brier |
|---|---|---|
| **Final: logistic(age, priors, priors/yr)** | **0.7381** | **0.2066** |
| B2: logistic(age, priors) | 0.7392 | 0.2093 |
| B3: COMPAS decile | 0.7131 | 0.2123 |

**Plainly stated: the final model's gain over B2 is negligible.** Test AUC is a hair *lower* (−0.001, pure noise); the Brier improvement (−0.003) means marginally better-calibrated probabilities but identical ranking ability. The honest conclusion is that age and priors carry essentially all the predictive signal, and the priors-rate transform adds only a small calibration refinement. Both comfortably beat COMPAS.

Calibration on test is good across the probability range (`plots/calibration_final_test.png`).

**Decision-threshold analysis** (test set; "FP among flagged" = share of detained people who would *not* have recidivated):

| Detain top… | n flagged | Prob. cutoff | FP among flagged | Recidivists caught |
|---|---|---|---|---|
| 10% | 124 | 0.753 | **18.5%** | 18.0% |
| 20% | 247 | 0.608 | **21.9%** | 34.3% |
| 30% | 370 | 0.501 | **25.1%** | 49.3% |

Even at the strictest threshold, roughly 1 in 5 people flagged would not have been rearrested; to catch half of eventual recidivists you must accept that 1 in 4 of those detained are false positives. This is the irreducible cost of acting on a ~0.74-AUC instrument. (`plots/threshold_analysis_test.png`)

## Phase 4 — Audit

Subgroup metrics at the top-20% threshold (test set). FPR/FNR are the standard conditional rates (FPR = flagged among true non-recidivists; FNR = missed among true recidivists). **Race and sex were never used as model inputs.**

| Group | n | Base rate | AUC | FPR | FNR | Flag rate |
|---|---|---|---|---|---|---|
| African-American | 641 | 0.526 | 0.760 | **12.5%** | 55.2% | 29.5% |
| Caucasian | 427 | 0.377 | 0.685 | **3.8%** | 81.4% | 9.4% |
| Hispanic | 94 | 0.479 | 0.670 | 8.2% | 73.3% | 17.0% |
| Other | 65 | 0.277 | 0.690 | 4.3% | 83.3% | 7.7% |
| Male | 1,002 | 0.476 | 0.746 | 9.1% | 62.7% | 22.6% |
| Female | 233 | 0.365 | 0.665 | 4.1% | 78.8% | 10.3% |

**This replicates the ProPublica/Northpointe dispute exactly**, despite race never entering the model. The false-positive rate for African-American defendants (12.5%) is over three times the Caucasian rate (3.8%), while the false-negative pattern runs the other way. This is not (only) a modeling flaw — it is arithmetic: when base rates differ across groups (52.6% vs 37.7% here) and the model is calibrated within groups, **equal calibration and equal error rates are mathematically incompatible** (Chouldechova 2017; Kleinberg et al. 2016). A calibrated score applied with a single threshold will flag the higher-base-rate group more often and therefore generate more false positives within it. Any deployment must decide which fairness criterion to satisfy, because it cannot satisfy all of them. Small samples for Hispanic (n=94) and Other (n=65) groups make those estimates unstable.

**Interpretability** (standardized logistic coefficients): priors_per_year **+0.563**, priors_count **+0.392**, age **−0.357**. The entire model is "more priors, accumulated faster, at a younger age → higher risk." Share of signal from age and priors alone: **~100% of AUC** (B2 matches the final model's 0.739 within noise) and ~99% of Brier skill relative to the base-rate floor.

## Rounds 2–2b — pushing past the plateau (iterations 7–15)

At the user's request the loop was continued with the parsimony rule dropped (select on best validation Brier) and a wider hypothesis space, drawing features the first round never used: pretrial jail length-of-stay, charge-type categories parsed from the free-text charge descriptions, log-priors, joint "stacked" small features, and full kitchen-sink models. Full log in `iteration_log.md`; code in `round2.py` / `round2b.py`.

**Round 2** (iterations 7–11, train/validation only):

- **Iteration 7 — kept**: log jail length-of-stay improved validation AUC beyond 1 SE (0.7308 → 0.7411). Pretrial detention length proxies charge seriousness and the booking judge's own risk read.
- Iterations 8–11 each nudged validation Brier down (charge categories 0.2041, stacked small features 0.2038, log-priors 0.2036, and a C-tuned **L2 logistic on all 16 features at 0.2016**) but every step fell inside the 1-SE noise gate, so the streak rule fired.
- Selecting on raw best validation Brier anyway, the kitchen-sink L2 logistic (val Brier 0.2016, AUC 0.7489) was evaluated on the test set — **the project's second and final test-set touch** (all tuning stayed on train/validation):

| Model | Val Brier | Test AUC | Test Brier |
|---|---|---|---|
| Round 1: logistic(age, priors, priors/yr) | 0.2081 | 0.7381 | 0.2066 |
| Round 2: L2 kitchen sink (16 features) | 0.2016 | 0.7343 | 0.2069 |

The 0.0065 validation Brier advantage **vanished on test** — the bigger model is, if anything, marginally worse. The validation set (n=1,235) had been used to adjudicate 11 decisions by this point; its apparent gains were partly fit to that particular sample. This is the textbook reason the noise gate existed.

**Round 2b** (iterations 12–15, validation only — test never touched): the moves the streak rule had cut off, judged against the kitchen-sink reference (val Brier 0.2016, CV SE 0.0042):

| # | Move | Val Brier | Val AUC | Verdict |
|---|---|---|---|---|
| 12 | LightGBM, all 16 features, wide CV grid | 0.2036 | 0.7453 | worse |
| 13 | Isotonic-calibrated kitchen sink | 0.2017 | 0.7487 | tie |
| 14 | Soft-vote ensemble (logistic + GBM) | 0.2016 | 0.7500 | tie |
| 15 | + COMPAS decile score as a feature | **0.1992** | **0.7557** | within noise (ΔBrier −0.0024 < 1 SE) |

The most interesting probe is #15: the commercial COMPAS decile — which encodes a 137-item questionnaire (substance abuse, employment, associates, attitudes) unavailable in our features — adds the largest single validation gain of the entire project, hinting at modest incremental signal beyond criminal history. But it stays inside the noise gate, round 2 just demonstrated that validation gains of this size do not transfer, and a model that *requires* COMPAS as an input cannot serve as a COMPAS replacement. Not adopted; no third test touch taken.

**Conclusion of the extended search**: across 15 iterations, 17 candidate features, four model classes, hyperparameter tuning, recalibration, and ensembling, held-out test performance never moved from **AUC ≈ 0.74, Brier ≈ 0.207**. The recommended deployment model remains the round-1 three-variable logistic regression (age, priors, priors-per-year): nothing more complex earned its keep, and on test nothing beat it. The plateau is not an artifact of a timid search — it is the information ceiling of administrative criminal-history data in this dataset.

## Limitations

1. **The target is rearrest, not reoffending.** Arrests reflect policing intensity and geography as well as behavior; differential enforcement contaminates both the label and, through `priors_count`, the features. The model predicts contact with the criminal-justice system.
2. **One county, one period.** Broward County, FL, 2013–2014 screenings. Base rates, charging practices, and policing have changed; transport to any other jurisdiction or era is unvalidated.
3. **Static snapshot.** Features are frozen at screening; the model cannot observe desistance in progress (treatment, employment, aging out) and will keep penalizing a stale record.
4. **Discrimination ceiling.** AUC ≈ 0.74 means substantial irreducible overlap between recidivists and non-recidivists; the threshold table quantifies the human cost of pretending otherwise.
5. **Incompatible fairness criteria.** With unequal base rates, no single-threshold calibrated tool can equalize FPR/FNR across groups. That choice is normative and belongs to policymakers, not the model.

## Deliverables

- `pipeline.py` — end-to-end reproducible round-1 pipeline (download → clean → split → baselines → iteration loop → test evaluation → audit)
- `round2.py`, `round2b.py` — extended search (iterations 7–15)
- `iteration_log.md` — complete 15-iteration history with hypotheses, metrics, and decisions
- `final_report.md` — this document
- `final_model.joblib` — recommended model (3-variable logistic); `final_model_round2.joblib` — round-2 kitchen-sink model; `results.json`, `results_round2.json`, `results_round2b.json` — all metrics
- `plots/` — calibration plots (B2/B3 validation; round-1 final, B2, and round-2 model on test) and threshold analysis
