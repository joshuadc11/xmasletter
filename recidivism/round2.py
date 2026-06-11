"""
Round 2: continue the iteration loop past the round-1 plateau, selecting on
BEST validation Brier (parsimony rule dropped per user instruction).

New hypothesis space: jail length-of-stay, charge-type categories parsed from
c_charge_desc, stacked small features, kitchen-sink L2 logistic, wider-grid
LightGBM, calibrated GBM, logistic+GBM ensemble, log-priors, and the COMPAS
decile as an incremental-validity probe.

Same cleaning, same seeded 60/20/20 split as round 1 (verified identical).
All tuning on train/validation only. The final winner is evaluated on the
test set — disclosed as the SECOND test-set touch of the project.

Outputs: appends to iteration_log.md; writes results_round2.json,
final_model_round2.joblib, plots/calibration_round2_test.png.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier

from pipeline import (DATA, HERE, SEED, calibration_plot, cv_metrics, evaluate,
                      make_logistic)


# ----------------------------------------------------------------------------
# Extended load: same ProPublica filter, more engineered features.
# ----------------------------------------------------------------------------
def load_extended():
    raw = pd.read_csv(DATA)
    df = raw[
        (raw["days_b_screening_arrest"] >= -30)
        & (raw["days_b_screening_arrest"] <= 30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"].notna())
        & (raw["score_text"] != "N/A")
    ].copy()

    df["juv_total"] = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    df["priors_per_year"] = df["priors_count"] / np.maximum(df["age"] - 18, 1)
    df["charge_felony"] = (df["c_charge_degree"] == "F").astype(int)
    df["sex_male"] = (df["sex"] == "Male").astype(int)
    df["age_x_priors"] = df["age"] * df["priors_count"]
    df["log_priors"] = np.log1p(df["priors_count"])

    # Pretrial jail length of stay (days), available at screening time.
    jin = pd.to_datetime(df["c_jail_in"], errors="coerce")
    jout = pd.to_datetime(df["c_jail_out"], errors="coerce")
    stay = (jout - jin).dt.total_seconds() / 86400.0
    df["log_jail_days"] = np.log1p(stay.clip(lower=0)).fillna(0.0)
    df["jail_missing"] = stay.isna().astype(int)

    # Charge category from free-text description (keyword rules).
    desc = df["c_charge_desc"].fillna("").str.lower()
    df["chg_violent"] = desc.str.contains(
        "batt|assault|violen|murder|manslaughter|robbery|weapon|firearm|"
        "carjack|kidnap|abuse|stalk|resist").astype(int)
    df["chg_drug"] = desc.str.contains(
        "cocaine|cannabis|drug|heroin|meth|mdma|oxyc|controlled sub|"
        "traffick|deliver|pos").astype(int)
    df["chg_property"] = desc.str.contains(
        "theft|burgl|larc|stolen|fraud|forg|trespass|damage|deal in").astype(int)
    df["chg_driving"] = desc.str.contains(
        "driving|license|dui|d\\.u\\.i|veh").astype(int)

    return df.reset_index(drop=True)


def main():
    df = load_extended()
    y = df["two_year_recid"]
    idx_trval, idx_test = train_test_split(
        df.index, test_size=0.20, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=0.25, stratify=y.loc[idx_trval], random_state=SEED)
    train, val, test = df.loc[idx_train], df.loc[idx_val], df.loc[idx_test]
    y_tr, y_va, y_te = (d["two_year_recid"] for d in (train, val, test))
    assert len(train) == 3702 and len(val) == 1235 and len(test) == 1235

    # Round-1 champion as starting point.
    champ_feats = ["age", "priors_count", "priors_per_year"]
    champ_builder = lambda f: make_logistic(f)
    champ_model = champ_builder(champ_feats)
    champ_cv = cv_metrics(champ_model, train[champ_feats], y_tr)
    champ_model.fit(train[champ_feats], y_tr)
    champ_val = evaluate(y_va, champ_model.predict_proba(val[champ_feats])[:, 1])
    print(f"Champion (round 1): brier={champ_val['brier']:.4f} "
          f"auc={champ_val['auc']:.4f}")

    state = {
        "champ_feats": champ_feats, "champ_builder": champ_builder,
        "champ_val": champ_val, "champ_cv": champ_cv,
        "champ_name": "round-1 winner: logistic(age, priors, priors/yr)",
        "best": None,  # best-by-val-Brier across ALL round-2 candidates
        "streak": 0, "iter": 6,  # continue numbering from round 1
        "log": [], "history": [],
    }

    def try_candidate(hypothesis, change, feats, builder):
        state["iter"] += 1
        model = builder(feats)
        cv = cv_metrics(model, train[feats], y_tr)
        model.fit(train[feats], y_tr)
        vm = evaluate(y_va, model.predict_proba(val[feats])[:, 1])

        se_b = max(cv["cv_brier_se"], state["champ_cv"]["cv_brier_se"])
        se_a = max(cv["cv_auc_se"], state["champ_cv"]["cv_auc_se"])
        cv_, vm_ = state["champ_cv"], state["champ_val"]
        keep, reason = False, ""
        if vm["brier"] < vm_["brier"] - se_b:
            keep, reason = True, "Brier improved beyond 1 SE"
        elif (vm["brier"] < vm_["brier"] + se_b and vm["auc"] > vm_["auc"] + se_a):
            keep, reason = True, "AUC improved beyond 1 SE, Brier within noise"
        else:
            reason = (f"no gain beyond noise (ΔBrier={vm['brier']-vm_['brier']:+.4f} "
                      f"vs SE={se_b:.4f}, ΔAUC={vm['auc']-vm_['auc']:+.4f} "
                      f"vs SE={se_a:.4f}) -> reverted")

        if state["best"] is None or vm["brier"] < state["best"]["val"]["brier"]:
            state["best"] = {"name": change, "feats": list(feats),
                             "builder": builder, "val": vm, "cv": cv}
        state["history"].append({"name": change, "val": vm, "cv": cv, "kept": keep})
        state["log"].append(
            f"| {state['iter']} | {hypothesis} | {change} | {vm['brier']:.4f} | "
            f"{vm['auc']:.4f} | {cv['cv_brier_mean']:.4f}±{cv['cv_brier_se']:.4f} | "
            f"{'KEPT — ' + reason if keep else 'REVERTED — ' + reason} |")
        print(f"Iter {state['iter']}: {change[:64]} brier={vm['brier']:.4f} "
              f"auc={vm['auc']:.4f} keep={keep}")
        if keep:
            state.update(champ_feats=list(feats), champ_builder=builder,
                         champ_val=vm, champ_cv=cv, champ_name=change, streak=0)
        else:
            state["streak"] += 1
        return keep

    def champ_plus(extra):
        return state["champ_feats"] + [f for f in extra
                                       if f not in state["champ_feats"]]

    ALL_FEATS = ["age", "priors_count", "priors_per_year", "log_priors",
                 "juv_fel_count", "juv_misd_count", "juv_other_count",
                 "charge_felony", "sex_male", "age_x_priors",
                 "log_jail_days", "jail_missing",
                 "chg_violent", "chg_drug", "chg_property", "chg_driving"]

    # ---- candidate sequence (one hypothesis per iteration) ----
    moves = []
    moves.append((
        "Pretrial jail length-of-stay proxies both charge seriousness and "
        "judicial risk assessment at booking; longer stays should predict "
        "recidivism beyond charge degree.",
        "Add log_jail_days + jail_missing",
        lambda: (champ_plus(["log_jail_days", "jail_missing"]),
                 state["champ_builder"])))
    moves.append((
        "Offense TYPE matters: drug and property offending have the highest "
        "repeat rates; violent index offenses historically predict lower "
        "general recidivism.",
        "Add charge-category dummies (violent/drug/property/driving)",
        lambda: (champ_plus(["chg_violent", "chg_drug", "chg_property",
                             "chg_driving"]), state["champ_builder"])))
    moves.append((
        "Round-1 features that individually fell just under the noise gate "
        "(charge degree, sex, juvenile record) may clear it jointly — small "
        "real effects stack.",
        "Add charge_felony + sex_male + juv_total together",
        lambda: (champ_plus(["charge_felony", "sex_male", "juv_total"]),
                 state["champ_builder"])))
    moves.append((
        "Marginal deterrent information per additional prior shrinks; "
        "log(1+priors) should fit the diminishing-returns shape better than "
        "the linear count.",
        "Add log_priors",
        lambda: (champ_plus(["log_priors"]), state["champ_builder"])))

    def l2_kitchen():
        base = make_logistic(ALL_FEATS, penalty="l2", C=1.0)
        gs = GridSearchCV(base, {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
                          scoring="neg_brier_score",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        gs.fit(train[ALL_FEATS], y_tr)
        C = gs.best_params_["clf__C"]
        return (ALL_FEATS,
                lambda f, C=C: make_logistic(f, penalty="l2", C=C)), f"C={C}"
    moves.append((
        "With 16 candidate features, L2 shrinkage can exploit many weak "
        "signals jointly while controlling variance — the 'kitchen sink, "
        "regularized' hypothesis.",
        "L2 logistic on all 16 features, C tuned by CV", l2_kitchen))

    def gbm_kitchen():
        grid = {"num_leaves": [7, 15, 31], "learning_rate": [0.02, 0.05, 0.1],
                "n_estimators": [200, 500], "min_child_samples": [30, 60],
                "colsample_bytree": [0.7, 1.0]}
        gs = GridSearchCV(LGBMClassifier(random_state=SEED, verbose=-1),
                          grid, scoring="neg_brier_score", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        gs.fit(train[ALL_FEATS], y_tr)
        bp = gs.best_params_
        return (ALL_FEATS,
                lambda f, bp=bp: LGBMClassifier(random_state=SEED, verbose=-1,
                                                **bp)), str(bp)
    moves.append((
        "Round-1 GBM only saw 3 features; with the full 16-feature space and "
        "a wider grid, boosting has interactions to exploit.",
        "LightGBM on all 16 features, wider CV grid", gbm_kitchen))

    def calibrated_best_gbm():
        # Calibrate whatever the current champion is.
        b = state["champ_builder"]
        return (list(state["champ_feats"]),
                lambda f, b=b: CalibratedClassifierCV(b(f), method="isotonic",
                                                      cv=5)), ""
    moves.append((
        "If the champion is now a tree ensemble, isotonic recalibration "
        "should recover Brier lost to miscalibrated leaves.",
        "Isotonic-calibrate current champion", calibrated_best_gbm))

    def ensemble():
        # Both members receive ALL_FEATS; the logistic pipeline's
        # ColumnTransformer selects its own columns by name.
        lr_feats = ["age", "priors_count", "priors_per_year"]
        return (ALL_FEATS, lambda f: VotingClassifier(
            [("lr", make_logistic(lr_feats)),
             ("gbm", LGBMClassifier(random_state=SEED, verbose=-1, num_leaves=7,
                                    learning_rate=0.03, n_estimators=100,
                                    min_child_samples=50))],
            voting="soft")), ""
    moves.append((
        "Logistic and GBM make partially uncorrelated errors; averaging "
        "their probabilities reduces variance (classic ensembling).",
        "Soft-vote ensemble: round-1 logistic + tuned GBM", ensemble))

    moves.append((
        "Incremental-validity probe: does the commercial COMPAS decile carry "
        "any signal not already in age/priors? (Literature says little.)",
        "Add decile_score to champion",
        lambda: (champ_plus(["decile_score"]), state["champ_builder"])))

    for hypothesis, change, make in moves:
        if state["streak"] >= 4 or state["iter"] >= 25:
            break
        out = make()
        if isinstance(out[1], str):  # tuned moves return ((feats,builder),info)
            (feats, builder), info = out
            change = change + (f" -> {info}" if info else "")
        else:
            feats, builder = out
        try_candidate(hypothesis, change, feats, builder)

    stop = ("4 consecutive non-improvements" if state["streak"] >= 4
            else "candidate moves exhausted")

    # ---- selection: BEST validation Brier across all round-2 candidates,
    # round-1 champion included (parsimony rule dropped per user request).
    best = state["best"]
    if champ_val["brier"] <= best["val"]["brier"]:
        best = {"name": "round-1 winner: logistic(age, priors, priors/yr)",
                "feats": champ_feats, "builder": champ_builder,
                "val": champ_val, "cv": champ_cv}
    print(f"\nRound-2 selection (best val Brier): {best['name']} "
          f"brier={best['val']['brier']:.4f} auc={best['val']['auc']:.4f}")

    # ---- final test evaluation (SECOND test-set touch, disclosed) ----
    trval = pd.concat([train, val])
    y_trval = trval["two_year_recid"]
    final = best["builder"](best["feats"])
    final.fit(trval[best["feats"]], y_trval)
    p_test = final.predict_proba(test[best["feats"]])[:, 1]
    test_m = evaluate(y_te, p_test)
    calibration_plot(y_te, p_test, "Round-2 best model (test)",
                     "calibration_round2_test.png")
    print("TEST (round-2 best):", test_m)

    # Subgroup audit at top-20% for the new model.
    k20 = int(round(len(test) * 0.20))
    cut20 = np.sort(p_test)[::-1][k20 - 1]
    flag = p_test >= cut20
    yv = y_te.values
    audit = []
    from sklearn.metrics import roc_auc_score
    for col, vals in (("race", test["race"].value_counts().index),
                      ("sex", ["Male", "Female"])):
        for g in vals:
            m = (test[col] == g).values
            if m.sum() < 50:
                continue
            yt, fl = yv[m], flag[m]
            tn = ((yt == 0) & ~fl).sum(); fp = ((yt == 0) & fl).sum()
            fn = ((yt == 1) & ~fl).sum(); tp = ((yt == 1) & fl).sum()
            audit.append({"group": f"{col}={g}", "n": int(m.sum()),
                          "base_rate": float(yt.mean()),
                          "auc": float(roc_auc_score(yt, p_test[m])),
                          "fpr": float(fp / (fp + tn)),
                          "fnr": float(fn / (fn + tp)),
                          "flag_rate": float(fl.mean())})
            print(" ", audit[-1])

    # Thresholds for the new model.
    thresh = []
    order = np.argsort(-p_test)
    for pct in (10, 20, 30):
        k = int(round(len(test) * pct / 100))
        fl = order[:k]
        thresh.append({"top_pct": pct, "n_flagged": k,
                       "prob_cutoff": float(p_test[order[k - 1]]),
                       "fp_rate_among_flagged": float(1 - y_te.iloc[fl].mean()),
                       "share_of_recidivists_caught":
                           float(y_te.iloc[fl].sum() / y_te.sum())})

    # ---- persist ----
    with open(os.path.join(HERE, "iteration_log.md"), "a") as fh:
        fh.write("\n\n## Round 2 (parsimony rule dropped; select best val Brier)\n\n")
        fh.write("| # | Hypothesis | Change | Val Brier | Val AUC | "
                 "CV Brier (±SE) | Decision |\n")
        fh.write("|---|------------|--------|-----------|---------|"
                 "----------------|----------|\n")
        fh.write("\n".join(state["log"]) + "\n\n")
        fh.write(f"**Stopped: {stop}.** Selected (best val Brier): "
                 f"**{best['name']}** (val Brier={best['val']['brier']:.4f}, "
                 f"AUC={best['val']['auc']:.4f}).\n")

    out = {"round2_iterations": state["history"],
           "selected": {"name": best["name"], "features": best["feats"],
                        "val": best["val"]},
           "test_second_touch": test_m,
           "thresholds": thresh,
           "subgroup_audit_top20": audit,
           "note": ("This is the project's SECOND evaluation on the held-out "
                    "test set (first was round 1). All tuning remained on "
                    "train/validation.")}
    if hasattr(final, "feature_importances_"):
        out["feature_importances"] = dict(
            zip(best["feats"], [float(v) for v in final.feature_importances_]))
    elif hasattr(final, "named_steps") and hasattr(
            final.named_steps.get("clf", None), "coef_"):
        names = final.named_steps["pre"].get_feature_names_out()
        out["coefficients"] = dict(zip([str(n) for n in names],
                                       [float(c) for c in
                                        final.named_steps["clf"].coef_[0]]))
    with open(os.path.join(HERE, "results_round2.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    joblib.dump(final, os.path.join(HERE, "final_model_round2.joblib"))
    print("Done. Wrote results_round2.json, final_model_round2.joblib, "
          "appended iteration_log.md")


if __name__ == "__main__":
    main()
