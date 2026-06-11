# Iteration Log

Champion starts at baseline B2 (logistic: age + priors_count). Val Brier=0.2109, AUC=0.7242; CV Brier=0.2130±0.0017 (SE).

Decision rule: keep if validation Brier improves by more than 1 CV standard error (primary), or Brier within noise AND validation AUC improves by more than 1 CV SE (secondary).

| # | Hypothesis | Change | Val Brier | Val AUC | CV Brier (±SE) | Decision |
|---|------------|--------|-----------|---------|----------------|----------|
| 1 | Juvenile record signals early criminal-career onset, a strong predictor of persistence (Moffitt's life-course taxonomy), beyond adult priors. | Add juv_fel_count, juv_misd_count, juv_other_count | 0.2105 | 0.7254 | 0.2131±0.0018 | REVERTED — no gain beyond noise (ΔBrier=-0.0004 vs SE=0.0018, ΔAUC=+0.0011 vs SE=0.0056) -> reverted |
| 2 | Prior offense RATE (priors per adult year) separates high-rate offenders from older offenders with long but slow careers; raw count conflates rate with exposure time. | Add priors_per_year | 0.2081 | 0.7308 | 0.2101±0.0023 | KEPT — Brier improved beyond 1 SE |
| 3 | The age-crime curve is sharply nonlinear (steep desistance after early 20s, flattening later); a linear age term misses this. | Replace linear age with cubic spline (5 knots) | 0.2088 | 0.7282 | 0.2086±0.0034 | REVERTED — no gain beyond noise (ΔBrier=+0.0007 vs SE=0.0034, ΔAUC=-0.0026 vs SE=0.0090) -> reverted |
| 4 | Felony vs misdemeanor index charge proxies offense seriousness, which correlates with criminal propensity. | Add charge_felony | 0.2067 | 0.7344 | 0.2096±0.0021 | REVERTED — no gain beyond noise (ΔBrier=-0.0014 vs SE=0.0023, ΔAUC=+0.0036 vs SE=0.0052) -> reverted |
| 5 | Gradient boosting can capture interactions/nonlinearities the linear model misses, if any remain. | LightGBM, small grid tuned by 5-fold CV (neg brier) -> best {'learning_rate': 0.03, 'min_child_samples': 50, 'n_estimators': 100, 'num_leaves': 7} | 0.2076 | 0.7301 | 0.2081±0.0034 | REVERTED — no gain beyond noise (ΔBrier=-0.0006 vs SE=0.0034, ΔAUC=-0.0008 vs SE=0.0094) -> reverted |
| 6 | Tree ensembles are often miscalibrated; isotonic calibration should improve Brier if the current champion is a GBM, and Platt scaling can sharpen a logistic champion. | CalibratedClassifierCV(champion, isotonic, cv=5) | 0.2075 | 0.7310 | 0.2099±0.0023 | REVERTED — no gain beyond noise (ΔBrier=-0.0006 vs SE=0.0023, ΔAUC=+0.0002 vs SE=0.0054) -> reverted |

**Stopped after 6 iterations: 4 consecutive non-improvements.**

**Parsimony rule**: best val Brier=0.2067 (SE≈0.0023); 5 configs within 1 SE; simplest selected: **Add priors_per_year** (val Brier=0.2081, AUC=0.7308).


## Round 2 (parsimony rule dropped; select best val Brier)

| # | Hypothesis | Change | Val Brier | Val AUC | CV Brier (±SE) | Decision |
|---|------------|--------|-----------|---------|----------------|----------|
| 7 | Pretrial jail length-of-stay proxies both charge seriousness and judicial risk assessment at booking; longer stays should predict recidivism beyond charge degree. | Add log_jail_days + jail_missing | 0.2054 | 0.7411 | 0.2082±0.0031 | KEPT — AUC improved beyond 1 SE, Brier within noise |
| 8 | Offense TYPE matters: drug and property offending have the highest repeat rates; violent index offenses historically predict lower general recidivism. | Add charge-category dummies (violent/drug/property/driving) | 0.2041 | 0.7423 | 0.2075±0.0038 | REVERTED — no gain beyond noise (ΔBrier=-0.0013 vs SE=0.0038, ΔAUC=+0.0012 vs SE=0.0102) -> reverted |
| 9 | Round-1 features that individually fell just under the noise gate (charge degree, sex, juvenile record) may clear it jointly — small real effects stack. | Add charge_felony + sex_male + juv_total together | 0.2038 | 0.7436 | 0.2080±0.0033 | REVERTED — no gain beyond noise (ΔBrier=-0.0016 vs SE=0.0033, ΔAUC=+0.0025 vs SE=0.0087) -> reverted |
| 10 | Marginal deterrent information per additional prior shrinks; log(1+priors) should fit the diminishing-returns shape better than the linear count. | Add log_priors | 0.2036 | 0.7461 | 0.2077±0.0031 | REVERTED — no gain beyond noise (ΔBrier=-0.0018 vs SE=0.0031, ΔAUC=+0.0050 vs SE=0.0083) -> reverted |
| 11 | With 16 candidate features, L2 shrinkage can exploit many weak signals jointly while controlling variance — the 'kitchen sink, regularized' hypothesis. | L2 logistic on all 16 features, C tuned by CV -> C=0.3 | 0.2016 | 0.7488 | 0.2063±0.0042 | REVERTED — no gain beyond noise (ΔBrier=-0.0037 vs SE=0.0042, ΔAUC=+0.0077 vs SE=0.0113) -> reverted |

**Stopped: 4 consecutive non-improvements.** Selected (best val Brier): **L2 logistic on all 16 features, C tuned by CV -> C=0.3** (val Brier=0.2016, AUC=0.7488).

## Round 2b (remaining moves; validation only — test NOT touched)

Reference: L2 kitchen sink val Brier=0.2016, AUC=0.7488 (CV SE=0.0042).

| # | Hypothesis | Change | Val Brier | Val AUC | CV Brier (±SE) | vs reference |
|---|------------|--------|-----------|---------|----------------|--------------|
| 12 | Boosting with 16 features has interactions to exploit that the 3-feature round-1 GBM could not. | LightGBM all-16, wide grid -> {'colsample_bytree': 1.0, 'learning_rate': 0.02, 'min_child_samples': 60, 'n_estimators': 150, 'num_leaves': 7} | 0.2036 | 0.7453 | 0.2071±0.0036 | within noise (ΔBrier=+0.0019) |
| 13 | Isotonic recalibration can fix any residual miscalibration in the regularized logistic. | Isotonic-calibrate L2 kitchen sink | 0.2017 | 0.7487 | 0.2069±0.0040 | within noise (ΔBrier=+0.0001) |
| 14 | Logistic and GBM make partially uncorrelated errors; averaging probabilities reduces variance. | Soft-vote: L2 kitchen sink + tuned GBM | 0.2016 | 0.7500 | 0.2055±0.0039 | within noise (ΔBrier=-0.0000) |
| 15 | Does the commercial COMPAS decile carry signal not already in the 16 features? (Literature: little.) | Add decile_score to L2 kitchen sink | 0.1992 | 0.7557 | 0.2032±0.0044 | within noise (ΔBrier=-0.0024) |
