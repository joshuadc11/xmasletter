"""
Actuarial two-year recidivism model on the ProPublica COMPAS dataset.

Phases:
  1. Load/clean, 60/20/20 stratified split (test set locked), baselines B1-B3.
  2. Iterative improvement loop: one hypothesis per iteration, evaluated on the
     validation set with 5-fold CV noise estimates. Stop after 4 consecutive
     non-improvements or 25 iterations.
  3. Retrain winner on train+val, evaluate ONCE on test, threshold analysis.
  4. Subgroup audit (race, sex), interpretability, report inputs.

All randomness seeded (SEED=42). Outputs: iteration_log.md, results.json,
plots/*.png, final_model.joblib.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "compas-scores-two-years.csv")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

rng = np.random.RandomState(SEED)


# ----------------------------------------------------------------------------
# Phase 1a: load and clean
# ----------------------------------------------------------------------------
DATA_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
            "master/compas-scores-two-years.csv")


def load_and_clean():
    if not os.path.exists(DATA):
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, DATA)
    raw = pd.read_csv(DATA)
    n_raw = len(raw)
    df = raw[
        (raw["days_b_screening_arrest"] >= -30)
        & (raw["days_b_screening_arrest"] <= 30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"].notna())
        & (raw["score_text"] != "N/A")
    ].copy()

    # Engineered features (all row-wise transforms -> no leakage concerns).
    df["juv_total"] = (
        df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    )
    # Priors per year of adult exposure (age 18 onward, floor 1 year).
    df["priors_per_year"] = df["priors_count"] / np.maximum(df["age"] - 18, 1)
    df["charge_felony"] = (df["c_charge_degree"] == "F").astype(int)
    df["sex_male"] = (df["sex"] == "Male").astype(int)
    df["age_x_priors"] = df["age"] * df["priors_count"]

    keep = [
        "age", "priors_count", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "juv_total", "priors_per_year", "charge_felony",
        "sex_male", "age_x_priors", "decile_score", "race", "sex",
        "two_year_recid",
    ]
    df = df[keep].reset_index(drop=True)

    info = {
        "n_raw": int(n_raw),
        "n_clean": int(len(df)),
        "base_rate": float(df["two_year_recid"].mean()),
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
    }
    return df, info


# ----------------------------------------------------------------------------
# Metrics / plots
# ----------------------------------------------------------------------------
def evaluate(y_true, p):
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def calibration_plot(y_true, p, title, fname):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed recidivism rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------------
def make_logistic(features, age_spline=False, penalty=None, C=1.0):
    """Standardized logistic regression over `features`; optional age spline."""
    transformers = []
    other = [f for f in features if f != "age" or not age_spline]
    if age_spline and "age" in features:
        transformers.append(
            ("age_spline",
             Pipeline([("spline", SplineTransformer(n_knots=5, degree=3)),
                       ("sc", StandardScaler())]),
             ["age"]))
    transformers.append(("num", StandardScaler(), other))
    pre = ColumnTransformer(transformers)
    if penalty is None:
        clf = LogisticRegression(penalty=None, max_iter=2000, random_state=SEED)
    else:
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        clf = LogisticRegression(penalty=penalty, C=C, solver=solver,
                                 max_iter=2000, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def cv_metrics(model, X, y, n_splits=5):
    """5-fold CV on the training set -> mean and SE of brier/auc."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    briers, aucs = [], []
    for tr, te in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        briers.append(brier_score_loss(y.iloc[te], p))
        aucs.append(roc_auc_score(y.iloc[te], p))
    return {
        "cv_brier_mean": float(np.mean(briers)),
        "cv_brier_se": float(np.std(briers, ddof=1) / np.sqrt(n_splits)),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_se": float(np.std(aucs, ddof=1) / np.sqrt(n_splits)),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    results = {}
    df, info = load_and_clean()
    results["data"] = info
    print(f"Cleaned: {info['n_clean']} rows (from {info['n_raw']}); "
          f"base rate = {info['base_rate']:.3f}")

    y = df["two_year_recid"]
    # 60 / 20 / 20 stratified split. Test set locked until Phase 3.
    idx_trval, idx_test = train_test_split(
        df.index, test_size=0.20, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=0.25, stratify=y.loc[idx_trval], random_state=SEED)

    train, val, test = df.loc[idx_train], df.loc[idx_val], df.loc[idx_test]
    y_tr, y_va, y_te = (d["two_year_recid"] for d in (train, val, test))
    results["split"] = {"n_train": len(train), "n_val": len(val),
                        "n_test": len(test),
                        "base_rates": {"train": float(y_tr.mean()),
                                       "val": float(y_va.mean()),
                                       "test": float(y_te.mean())}}
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # ------------------------------------------------------------------
    # Phase 1b: baselines (validation set)
    # ------------------------------------------------------------------
    baselines = {}

    # B1: base-rate predictor.
    p_b1 = np.full(len(val), y_tr.mean())
    baselines["B1_base_rate"] = evaluate(y_va, p_b1)

    # B2: age + priors logistic (Dressel & Farid benchmark).
    b2_feats = ["age", "priors_count"]
    b2 = make_logistic(b2_feats)
    b2.fit(train[b2_feats], y_tr)
    p_b2 = b2.predict_proba(val[b2_feats])[:, 1]
    baselines["B2_age_priors_logit"] = evaluate(y_va, p_b2)
    calibration_plot(y_va, p_b2, "B2: age+priors logistic (val)",
                     "calibration_B2_val.png")

    # B3: COMPAS decile score. AUC is scale-free; for Brier/calibration map
    # deciles to probabilities via training-set recidivism rate per decile.
    decile_rate = train.groupby("decile_score")["two_year_recid"].mean()
    p_b3 = val["decile_score"].map(decile_rate).fillna(y_tr.mean()).values
    baselines["B3_compas_decile"] = {
        "auc_raw_decile": float(roc_auc_score(y_va, val["decile_score"])),
        **evaluate(y_va, p_b3),
    }
    calibration_plot(y_va, p_b3, "B3: COMPAS decile (val)",
                     "calibration_B3_val.png")

    results["baselines_val"] = baselines
    for k, v in baselines.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Phase 2: iteration loop
    # ------------------------------------------------------------------
    # Champion state starts at B2.
    champion = {
        "name": "B2: logistic(age, priors_count)",
        "features": b2_feats,
        "builder": lambda f: make_logistic(f),
        "complexity": 1,
    }
    champ_cv = cv_metrics(champion["builder"](champion["features"])
                          .set_params(), train[champion["features"]], y_tr)
    champ_val = evaluate(y_va, p_b2)
    history = []  # every evaluated config, for the parsimony rule
    history.append({"name": champion["name"], "val": champ_val,
                    "cv": champ_cv, "complexity": champion["complexity"],
                    "kept": True,
                    "features": list(champion["features"]),
                    "builder": champion["builder"]})

    log_lines = [
        "# Iteration Log",
        "",
        "Champion starts at baseline B2 (logistic: age + priors_count). "
        f"Val Brier={champ_val['brier']:.4f}, AUC={champ_val['auc']:.4f}; "
        f"CV Brier={champ_cv['cv_brier_mean']:.4f}±{champ_cv['cv_brier_se']:.4f} (SE).",
        "",
        "Decision rule: keep if validation Brier improves by more than 1 CV "
        "standard error (primary), or Brier within noise AND validation AUC "
        "improves by more than 1 CV SE (secondary).",
        "",
        "| # | Hypothesis | Change | Val Brier | Val AUC | CV Brier (±SE) | Decision |",
        "|---|------------|--------|-----------|---------|----------------|----------|",
    ]

    def gbm_builder_factory(features, params):
        def build(f):
            return LGBMClassifier(random_state=SEED, verbose=-1, **params)
        return build

    # Candidate moves: (hypothesis, change description, features, builder, complexity)
    def candidates(champ):
        f = champ["features"]
        cands = []
        cands.append((
            "Juvenile record signals early criminal-career onset, a strong "
            "predictor of persistence (Moffitt's life-course taxonomy), beyond "
            "adult priors.",
            "Add juv_fel_count, juv_misd_count, juv_other_count",
            f + ["juv_fel_count", "juv_misd_count", "juv_other_count"],
            champ["builder"],
            champ["complexity"] + 1, {}))
        cands.append((
            "Prior offense RATE (priors per adult year) separates high-rate "
            "offenders from older offenders with long but slow careers; raw "
            "count conflates rate with exposure time.",
            "Add priors_per_year",
            f + ["priors_per_year"],
            champ["builder"],
            champ["complexity"] + 1, {}))
        cands.append((
            "The age-crime curve is sharply nonlinear (steep desistance after "
            "early 20s, flattening later); a linear age term misses this.",
            "Replace linear age with cubic spline (5 knots)",
            f,
            lambda feats: make_logistic(feats, age_spline=True),
            champ["complexity"] + 1, {"spline": True}))
        cands.append((
            "Felony vs misdemeanor index charge proxies offense seriousness, "
            "which correlates with criminal propensity.",
            "Add charge_felony",
            f + ["charge_felony"],
            champ["builder"],
            champ["complexity"] + 1, {}))
        cands.append((
            "Gradient boosting can capture interactions/nonlinearities the "
            "linear model misses, if any remain.",
            "LightGBM, small grid tuned by 5-fold CV (neg brier)",
            f, "GBM_TUNE", champ["complexity"] + 3, {}))
        cands.append((
            "Tree ensembles are often miscalibrated; isotonic calibration "
            "should improve Brier if the current champion is a GBM, and Platt "
            "scaling can sharpen a logistic champion.",
            "CalibratedClassifierCV(champion, isotonic, cv=5)",
            f, "CALIBRATE", champ["complexity"] + 1, {}))
        cands.append((
            "Priors should matter more for young offenders (same count in "
            "fewer years at risk) — an age x priors interaction.",
            "Add age_x_priors",
            f + ["age_x_priors"],
            champ["builder"],
            champ["complexity"] + 1, {}))
        cands.append((
            "Sex is among the strongest demographic correlates of offending "
            "(male rates several times female) and is used in most actuarial "
            "tools.",
            "Add sex_male",
            f + ["sex_male"],
            champ["builder"],
            champ["complexity"] + 1, {}))
        cands.append((
            "With more features, L2 shrinkage should reduce variance without "
            "much bias; tune C by CV.",
            "L2-regularized logistic, C tuned over {0.01,0.1,1,10}",
            f, "L2_TUNE", champ["complexity"] + 1, {}))
        return cands

    no_improve_streak = 0
    iteration = 0
    cand_queue = candidates(champion)
    qi = 0
    requeued_after_keep = False

    while iteration < 25 and no_improve_streak < 4 and qi < len(cand_queue):
        hypothesis, change, feats, builder, complexity, extra = cand_queue[qi]
        qi += 1
        iteration += 1

        # Resolve special builders.
        if builder == "L2_TUNE":
            base = make_logistic(feats, age_spline=champion.get("spline", False),
                                 penalty="l2", C=1.0)
            gs = GridSearchCV(base, {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                              scoring="neg_brier_score",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            gs.fit(train[feats], y_tr)
            bestC = gs.best_params_["clf__C"]
            change += f" -> best C={bestC}"
            spline_flag = champion.get("spline", False)
            builder = (lambda fl, C=bestC, sp=spline_flag:
                       make_logistic(fl, age_spline=sp, penalty="l2", C=C))
        elif builder == "GBM_TUNE":
            grid = {"num_leaves": [7, 15, 31],
                    "learning_rate": [0.03, 0.1],
                    "n_estimators": [100, 300],
                    "min_child_samples": [20, 50]}
            gs = GridSearchCV(LGBMClassifier(random_state=SEED, verbose=-1),
                              grid, scoring="neg_brier_score",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            gs.fit(train[feats], y_tr)
            bp = gs.best_params_
            change += f" -> best {bp}"
            builder = (lambda fl, bp=bp:
                       LGBMClassifier(random_state=SEED, verbose=-1, **bp))
        elif builder == "CALIBRATE":
            inner_builder = champion["builder"]
            method = "isotonic"
            builder = (lambda fl, ib=inner_builder, m=method:
                       CalibratedClassifierCV(ib(fl), method=m, cv=5))
            change = f"CalibratedClassifierCV(champion, {method}, cv=5)"

        model = builder(feats)
        cv = cv_metrics(model, train[feats], y_tr)
        model.fit(train[feats], y_tr)
        p_val = model.predict_proba(val[feats])[:, 1]
        vm = evaluate(y_va, p_val)

        # Decision: 1-SE rule against current champion on validation Brier
        # (primary), AUC (secondary). Noise = max of the two configs' CV SEs.
        se_b = max(cv["cv_brier_se"], champ_cv["cv_brier_se"])
        se_a = max(cv["cv_auc_se"], champ_cv["cv_auc_se"])
        keep = False
        reason = ""
        if vm["brier"] < champ_val["brier"] - se_b:
            keep, reason = True, "Brier improved beyond 1 SE"
        elif (vm["brier"] < champ_val["brier"] + se_b
              and vm["auc"] > champ_val["auc"] + se_a):
            keep, reason = True, "AUC improved beyond 1 SE, Brier within noise"
        else:
            d_b = vm["brier"] - champ_val["brier"]
            d_a = vm["auc"] - champ_val["auc"]
            reason = (f"no gain beyond noise (ΔBrier={d_b:+.4f} vs SE={se_b:.4f}, "
                      f"ΔAUC={d_a:+.4f} vs SE={se_a:.4f}) -> reverted")

        history.append({"name": change, "val": vm, "cv": cv,
                        "complexity": complexity, "kept": keep,
                        "features": list(feats), "builder": builder})

        log_lines.append(
            f"| {iteration} | {hypothesis} | {change} | {vm['brier']:.4f} | "
            f"{vm['auc']:.4f} | {cv['cv_brier_mean']:.4f}±{cv['cv_brier_se']:.4f} | "
            f"{'KEPT — ' + reason if keep else 'REVERTED — ' + reason} |")
        print(f"Iter {iteration}: {change[:60]}... "
              f"brier={vm['brier']:.4f} auc={vm['auc']:.4f} keep={keep}")

        if keep:
            champion = {"name": change, "features": feats, "builder": builder,
                        "complexity": complexity,
                        "spline": extra.get("spline", champion.get("spline", False))}
            champ_cv, champ_val = cv, vm
            no_improve_streak = 0
            # Regenerate untried candidate move types against the new champion
            # (so later feature adds build on the kept features/model).
            tried_changes = {c.split("->")[0].strip()
                             for (_h, c, *_r) in cand_queue[:qi]}
            new_q = [c for c in candidates(champion)
                     if c[1].split("->")[0].strip() not in tried_changes]
            cand_queue = cand_queue[:qi] + new_q
        else:
            no_improve_streak += 1

    stop_reason = ("4 consecutive non-improvements"
                   if no_improve_streak >= 4 else
                   "candidate moves exhausted" if qi >= len(cand_queue)
                   else "25-iteration cap")
    log_lines += ["", f"**Stopped after {iteration} iterations: {stop_reason}.**"]

    # Parsimony rule: among ALL evaluated configs, pick the simplest whose
    # validation Brier is within 1 SE of the best validation Brier.
    best_brier = min(h["val"]["brier"] for h in history)
    best_se = champ_cv["cv_brier_se"]
    eligible = [h for h in history if h["val"]["brier"] <= best_brier + best_se]
    eligible.sort(key=lambda h: (h["complexity"], h["val"]["brier"]))
    winner = eligible[0]
    log_lines += [
        "",
        f"**Parsimony rule**: best val Brier={best_brier:.4f} (SE≈{best_se:.4f}); "
        f"{len(eligible)} configs within 1 SE; simplest selected: **{winner['name']}** "
        f"(val Brier={winner['val']['brier']:.4f}, AUC={winner['val']['auc']:.4f}).",
    ]
    print(f"Winner (parsimony): {winner['name']}")

    with open(os.path.join(HERE, "iteration_log.md"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")

    results["iterations"] = [
        {k: h[k] for k in ("name", "val", "cv", "complexity", "kept")}
        for h in history]
    results["winner"] = {"name": winner["name"], "features": winner["features"],
                         "val": winner["val"]}

    # ------------------------------------------------------------------
    # Phase 3: final evaluation — test set touched exactly once
    # ------------------------------------------------------------------
    trval = pd.concat([train, val])
    y_trval = trval["two_year_recid"]
    feats = winner["features"]

    final_model = winner["builder"](feats)
    final_model.fit(trval[feats], y_trval)
    p_test = final_model.predict_proba(test[feats])[:, 1]
    test_final = evaluate(y_te, p_test)
    calibration_plot(y_te, p_test, "Final model (test)",
                     "calibration_final_test.png")

    # B2 retrained on train+val for the side-by-side.
    b2f = make_logistic(b2_feats)
    b2f.fit(trval[b2_feats], y_trval)
    p_test_b2 = b2f.predict_proba(test[b2_feats])[:, 1]
    test_b2 = evaluate(y_te, p_test_b2)
    calibration_plot(y_te, p_test_b2, "B2 age+priors (test)",
                     "calibration_B2_test.png")

    # COMPAS decile on test for reference (mapped via train+val decile rates).
    decile_rate2 = trval.groupby("decile_score")["two_year_recid"].mean()
    p_test_b3 = test["decile_score"].map(decile_rate2).fillna(y_trval.mean()).values
    test_b3 = evaluate(y_te, p_test_b3)

    # Threshold analysis: top 10/20/30% predicted risk; FPR among flagged
    # = share of flagged who did NOT recidivate (1 - precision).
    thresh_rows = []
    for pct in (10, 20, 30):
        k = int(round(len(test) * pct / 100))
        order = np.argsort(-p_test)
        flagged = order[:k]
        cut = p_test[order[k - 1]]
        fp_among_flagged = float(1 - y_te.iloc[flagged].mean())
        recall = float(y_te.iloc[flagged].sum() / y_te.sum())
        thresh_rows.append({"top_pct": pct, "n_flagged": k,
                            "prob_cutoff": float(cut),
                            "fp_rate_among_flagged": fp_among_flagged,
                            "share_of_recidivists_caught": recall})
    results["test"] = {"final": test_final, "B2": test_b2,
                       "B3_compas": test_b3, "thresholds": thresh_rows}
    print("TEST final:", test_final, "| B2:", test_b2)

    # Threshold plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    pcts = [r["top_pct"] for r in thresh_rows]
    ax.bar([str(p) + "%" for p in pcts],
           [r["fp_rate_among_flagged"] for r in thresh_rows], color="#c44")
    ax.set_ylabel("False positives among flagged")
    ax.set_xlabel("Detention threshold (top % of predicted risk)")
    ax.set_title("Cost of flagging: share of flagged who did not recidivate")
    for i, r in enumerate(thresh_rows):
        ax.text(i, r["fp_rate_among_flagged"] + .01,
                f"{r['fp_rate_among_flagged']:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "threshold_analysis_test.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Phase 4: subgroup audit at top-20% threshold (test set)
    # ------------------------------------------------------------------
    k20 = int(round(len(test) * 0.20))
    cut20 = np.sort(p_test)[::-1][k20 - 1]
    flag = p_test >= cut20
    yv = y_te.values

    def group_audit(mask, label):
        if mask.sum() < 50:
            return None
        yt, fl, pp = yv[mask], flag[mask], p_test[mask]
        tn = ((yt == 0) & ~fl).sum(); fp = ((yt == 0) & fl).sum()
        fn = ((yt == 1) & ~fl).sum(); tp = ((yt == 1) & fl).sum()
        return {
            "group": label, "n": int(mask.sum()),
            "base_rate": float(yt.mean()),
            "auc": float(roc_auc_score(yt, pp)) if 0 < yt.mean() < 1 else None,
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
            "fnr": float(fn / (fn + tp)) if fn + tp else None,
            "flag_rate": float(fl.mean()),
        }

    audit = []
    for race in test["race"].value_counts().index:
        a = group_audit((test["race"] == race).values, f"race={race}")
        if a:
            audit.append(a)
    for sex in ("Male", "Female"):
        a = group_audit((test["sex"] == sex).values, f"sex={sex}")
        if a:
            audit.append(a)
    results["subgroup_audit_top20"] = audit
    for a in audit:
        print(" ", a)

    # Interpretability: coefficients (logistic) or importances (GBM).
    interp = {}
    fm = final_model
    if isinstance(fm, Pipeline) and hasattr(fm.named_steps.get("clf", None), "coef_"):
        names = fm.named_steps["pre"].get_feature_names_out()
        interp["type"] = "logistic_coefficients_standardized"
        interp["values"] = dict(zip([str(n) for n in names],
                                    [float(c) for c in fm.named_steps["clf"].coef_[0]]))
    elif hasattr(fm, "feature_importances_"):
        interp["type"] = "gbm_feature_importances"
        interp["values"] = dict(zip(feats,
                                    [float(v) for v in fm.feature_importances_]))
    elif isinstance(fm, CalibratedClassifierCV):
        interp["type"] = "calibrated_wrapper"
        interp["values"] = "see underlying estimator"
    # Share of signal from age+priors alone: B2 vs final on test.
    interp["age_priors_share"] = {
        "b2_test_auc": test_b2["auc"], "final_test_auc": test_final["auc"],
        "auc_gain_over_b2": test_final["auc"] - test_b2["auc"],
        "b2_test_brier": test_b2["brier"], "final_test_brier": test_final["brier"],
    }
    results["interpretability"] = interp

    joblib.dump(final_model, os.path.join(HERE, "final_model.joblib"))
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("Done. Wrote results.json, iteration_log.md, final_model.joblib, plots/")


if __name__ == "__main__":
    main()
