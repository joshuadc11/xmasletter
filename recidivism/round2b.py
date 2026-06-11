"""
Round 2b: evaluate the candidate moves the streak rule cut off in round 2
(wide-grid GBM, calibrated kitchen-sink logistic, ensemble, COMPAS-decile
probe). Validation only — the test set is NOT touched here. If nothing beats
the round-2 best (L2 kitchen sink, val Brier 0.2016) beyond noise, the
existing test evaluations stand.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier

from pipeline import HERE, SEED, cv_metrics, evaluate, make_logistic
from round2 import load_extended

ALL_FEATS = ["age", "priors_count", "priors_per_year", "log_priors",
             "juv_fel_count", "juv_misd_count", "juv_other_count",
             "charge_felony", "sex_male", "age_x_priors",
             "log_jail_days", "jail_missing",
             "chg_violent", "chg_drug", "chg_property", "chg_driving"]


def main():
    df = load_extended()
    y = df["two_year_recid"]
    idx_trval, idx_test = train_test_split(
        df.index, test_size=0.20, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=0.25, stratify=y.loc[idx_trval], random_state=SEED)
    train, val = df.loc[idx_train], df.loc[idx_val]
    y_tr, y_va = train["two_year_recid"], val["two_year_recid"]

    rows = []

    def run(num, hypothesis, change, feats, model):
        cv = cv_metrics(model, train[feats], y_tr)
        model.fit(train[feats], y_tr)
        vm = evaluate(y_va, model.predict_proba(val[feats])[:, 1])
        rows.append((num, hypothesis, change, vm, cv))
        print(f"Iter {num}: {change[:64]} brier={vm['brier']:.4f} "
              f"auc={vm['auc']:.4f} (cv {cv['cv_brier_mean']:.4f}"
              f"±{cv['cv_brier_se']:.4f})")
        return vm

    # Reference: round-2 best (L2 kitchen sink, C=0.3).
    ref = make_logistic(ALL_FEATS, penalty="l2", C=0.3)
    ref_cv = cv_metrics(ref, train[ALL_FEATS], y_tr)
    ref.fit(train[ALL_FEATS], y_tr)
    ref_vm = evaluate(y_va, ref.predict_proba(val[ALL_FEATS])[:, 1])
    print(f"Reference (L2 kitchen sink): brier={ref_vm['brier']:.4f} "
          f"auc={ref_vm['auc']:.4f}")

    # 12: wide-grid GBM on all 16 features.
    grid = {"num_leaves": [7, 15, 31], "learning_rate": [0.02, 0.05, 0.1],
            "n_estimators": [200, 500], "min_child_samples": [30, 60],
            "colsample_bytree": [0.7, 1.0]}
    gs = GridSearchCV(LGBMClassifier(random_state=SEED, verbose=-1), grid,
                      scoring="neg_brier_score", n_jobs=-1,
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    gs.fit(train[ALL_FEATS], y_tr)
    bp = gs.best_params_
    run(12, "Boosting with 16 features has interactions to exploit that the "
            "3-feature round-1 GBM could not.",
        f"LightGBM all-16, wide grid -> {bp}", ALL_FEATS,
        LGBMClassifier(random_state=SEED, verbose=-1, **bp))

    # 13: isotonic-calibrated kitchen-sink logistic.
    run(13, "Isotonic recalibration can fix any residual miscalibration in "
            "the regularized logistic.",
        "Isotonic-calibrate L2 kitchen sink",
        ALL_FEATS, CalibratedClassifierCV(
            make_logistic(ALL_FEATS, penalty="l2", C=0.3),
            method="isotonic", cv=5))

    # 14: soft-vote ensemble of L2 kitchen sink + tuned GBM.
    run(14, "Logistic and GBM make partially uncorrelated errors; averaging "
            "probabilities reduces variance.",
        "Soft-vote: L2 kitchen sink + tuned GBM",
        ALL_FEATS, VotingClassifier(
            [("lr", make_logistic(ALL_FEATS, penalty="l2", C=0.3)),
             ("gbm", LGBMClassifier(random_state=SEED, verbose=-1, **bp))],
            voting="soft"))

    # 15: incremental-validity probe — add COMPAS decile.
    run(15, "Does the commercial COMPAS decile carry signal not already in "
            "the 16 features? (Literature: little.)",
        "Add decile_score to L2 kitchen sink",
        ALL_FEATS + ["decile_score"],
        make_logistic(ALL_FEATS + ["decile_score"], penalty="l2", C=0.3))

    # Append to log.
    with open(os.path.join(HERE, "iteration_log.md"), "a") as fh:
        fh.write("\n## Round 2b (remaining moves; validation only — test NOT "
                 "touched)\n\n")
        fh.write(f"Reference: L2 kitchen sink val Brier="
                 f"{ref_vm['brier']:.4f}, AUC={ref_vm['auc']:.4f} "
                 f"(CV SE={ref_cv['cv_brier_se']:.4f}).\n\n")
        fh.write("| # | Hypothesis | Change | Val Brier | Val AUC | "
                 "CV Brier (±SE) | vs reference |\n")
        fh.write("|---|------------|--------|-----------|---------|"
                 "----------------|--------------|\n")
        for num, hyp, chg, vm, cv in rows:
            d = vm["brier"] - ref_vm["brier"]
            verdict = ("BEATS reference beyond 1 SE"
                       if d < -max(cv["cv_brier_se"], ref_cv["cv_brier_se"])
                       else f"within noise (ΔBrier={d:+.4f})")
            fh.write(f"| {num} | {hyp} | {chg} | {vm['brier']:.4f} | "
                     f"{vm['auc']:.4f} | {cv['cv_brier_mean']:.4f}"
                     f"±{cv['cv_brier_se']:.4f} | {verdict} |\n")

    with open(os.path.join(HERE, "results_round2b.json"), "w") as fh:
        json.dump({"reference_l2_kitchen_sink": ref_vm,
                   "candidates": [
                       {"iter": n, "change": c, "val": vm, "cv": cv}
                       for n, _h, c, vm, cv in rows]}, fh, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
