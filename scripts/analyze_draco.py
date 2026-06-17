"""
analyze_draco.py -- interpretability for DRACO: recover closed-form formulas from
the per-step traces (trace_ep*.csv) that train_draco.py / train_draco_craft.py
dump when agent.trace_every > 0.

Three questions, each a standalone result:

  1. ORDERING RULE   order = f(IP, next_demand, msg_in...)
        hypothesis: collapses to ~ max(0, S* - IP)  -> the learned policy IS an
        interpretable base-stock / order-up-to rule (Clark & Scarf 1960).
  2. MESSAGE DECODING msg_out_j = g(inv, backlog, on_order, next_demand, z...)
        hypothesis: a channel tracks the agent's demand estimate -> the mechanistic
        reason communication suppresses bullwhip (agents share demand, not panic).
  3. BELIEF MEANING   z_j = h(recent demand statistics)
        hypothesis: z encodes the demand regime -> confirms online adaptation.

Primary engine: PySR (symbolic regression; Cranmer 2023). If PySR is not
installed, falls back to (a) linear + degree-2 polynomial fits with R^2 and
(b) a mutual-information feature ranking, which still answer "what does it depend
on" even without a closed form. Also clusters the messages (do they form a
discrete protocol?) and prints a message<->state correlation table.

Usage:
  pip install pysr        # optional but recommended; also needs Julia (pysr installs it)
  python scripts/analyze_draco.py --trace weights_draco/run_draco_<id>/trace_ep5000.csv
  python scripts/analyze_draco.py --trace ".../trace_*.csv" --agent retailer --no-pysr
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    _HAS_PYSR = False

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _HAS_CLUSTER = True
except Exception:
    _HAS_CLUSTER = False


def load_traces(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no trace files match {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} rows from {len(files)} file(s); agents={sorted(df['agent'].unique())}")
    return df


def _pysr_fit(X, y, feature_names, label, niter=40):
    model = PySRRegressor(
        niterations=niter,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "relu", "max(x,0)" if False else "abs"],
        maxsize=22, model_selection="best", progress=False, verbosity=0,
        deterministic=True, random_state=0, procs=0, multithreading=False,
    )
    model.fit(np.asarray(X), np.asarray(y), variable_names=list(feature_names))
    yhat = model.predict(np.asarray(X))
    print(f"  [PySR] {label}:  {model.sympy()}")
    print(f"         R^2 = {r2_score(y, yhat):.3f}")


def _fallback_fit(X, y, feature_names, label):
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    lin = LinearRegression().fit(X, y)
    r2_lin = r2_score(y, lin.predict(X))
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    quad = LinearRegression().fit(Xp, y)
    r2_quad = r2_score(y, quad.predict(Xp))
    mi = mutual_info_regression(X, y, random_state=0)
    order = np.argsort(mi)[::-1]
    terms = "  ".join(f"{feature_names[i]}:{lin.coef_[i]:+.3f}" for i in order[:6])
    print(f"  [fallback] {label}")
    print(f"     linear R^2={r2_lin:.3f} | deg-2 R^2={r2_quad:.3f} | intercept={lin.intercept_:+.3f}")
    print(f"     coeffs (MI-ranked): {terms}")
    print(f"     MI ranking: " + ", ".join(f"{feature_names[i]}={mi[i]:.2f}" for i in order[:6]))


def fit(df, target, features, label, use_pysr):
    cols = [c for c in features if c in df.columns]
    sub = df[cols + [target]].dropna()
    if len(sub) < 50:
        print(f"  [skip] {label}: only {len(sub)} rows")
        return
    X, y = sub[cols].values, sub[target].values
    if use_pysr and _HAS_PYSR:
        try:
            _pysr_fit(X, y, cols, label)
            return
        except Exception as e:
            print(f"  [PySR failed -> fallback] {e}")
    _fallback_fit(X, y, cols, label)


def message_clustering(df, msg_cols):
    if not _HAS_CLUSTER or not msg_cols:
        return
    M = df[msg_cols].dropna().values
    if len(M) < 50:
        return
    print("\n[message protocol] do the continuous messages collapse to discrete clusters?")
    best = None
    for k in range(2, 7):
        try:
            lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(M)
            s = silhouette_score(M, lab)
            print(f"   k={k}: silhouette={s:.3f}")
            if best is None or s > best[1]:
                best = (k, s)
        except Exception:
            pass
    if best:
        print(f"   -> best k={best[0]} (silhouette {best[1]:.3f}). "
              f"{'high => a discrete protocol emerged' if best[1] > 0.5 else 'low => messages are continuous/graded'}")


def message_state_correlation(df, msg_cols, state_cols):
    cols_s = [c for c in state_cols if c in df.columns]
    cols_m = [c for c in msg_cols if c in df.columns]
    if not cols_s or not cols_m:
        return
    print("\n[message <-> state] |Pearson r| (which state variable each channel encodes):")
    hdr = "        " + "".join(f"{c:>12}" for c in cols_s)
    print(hdr)
    for m in cols_m:
        row = f"{m:>8}"
        for s in cols_s:
            sub = df[[m, s]].dropna()
            r = np.corrcoef(sub[m], sub[s])[0, 1] if len(sub) > 2 else np.nan
            row += f"{abs(r):>12.2f}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="path or glob to trace_ep*.csv")
    ap.add_argument("--agent", default=None, help="restrict to one echelon (e.g. retailer)")
    ap.add_argument("--no-pysr", action="store_true")
    args = ap.parse_args()

    df = load_traces(args.trace)
    if args.agent:
        df = df[df["agent"] == args.agent].copy()
        print(f"restricted to agent='{args.agent}': {len(df)} rows")
    use_pysr = not args.no_pysr
    if use_pysr and not _HAS_PYSR:
        print("note: PySR not installed -> using linear/poly/MI fallback. `pip install pysr` for closed-form laws.")

    z_cols = sorted([c for c in df.columns if c.startswith("z")], key=lambda s: int(s[1:]))
    msg_in_cols = sorted([c for c in df.columns if c.startswith("msg_in")], key=lambda s: int(s[6:]))
    msg_out_cols = sorted([c for c in df.columns if c.startswith("msg_out")], key=lambda s: int(s[7:]))
    state_cols = ["inv", "backlog", "on_order", "next_demand", "IP"]

    print("\n=== 1. ORDERING RULE :  order = f(IP, next_demand, msg_in) ===")
    fit(df, "order", ["IP", "next_demand"] + msg_in_cols, "order", use_pysr)
    print("\n    (also) base-stock target :  S_target = f(next_demand, z, msg_in)")
    fit(df, "S_target", ["next_demand"] + z_cols + msg_in_cols, "S_target", use_pysr)

    print("\n=== 2. MESSAGE DECODING :  msg_out_j = g(state, z) ===")
    for m in msg_out_cols:
        fit(df, m, state_cols + z_cols, m, use_pysr)

    print("\n=== 3. BELIEF MEANING :  z_j = h(state) ===")
    for z in z_cols[:4]:
        fit(df, z, state_cols, z, use_pysr)

    message_clustering(df, msg_out_cols)
    message_state_correlation(df, msg_out_cols, state_cols)
    print("\ndone. closed-form laws (or MI rankings) above are the interpretability results for the thesis.\n")


if __name__ == "__main__":
    main()