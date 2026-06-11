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
