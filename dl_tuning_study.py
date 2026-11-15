"""Give the neural networks their best shot: tune them, then test them honestly.

Usage:  python dl_tuning_study.py [--smoke]

This script reuses the notebook's own definitions (data loading, feature
engineering, the 80/20 split, the development and tuning subsamples, the
leakage-safe preprocessing and the Keras wrapper) by executing those cells from
thesis_churn_v2.ipynb, so the tuned network is evaluated on exactly the same
rows and folds as every other model in the study.

Protocol (the guard against overfitting is the separation of the three stages):
  1. SEARCH   Optuna (TPE, median pruning) over architecture and optimiser on the
              100k tuning sample, 3-fold CV, ROC-AUC. Early stopping on an inner
              validation split inside every fit.
  2. CONFIRM  The single best configuration is re-evaluated on the 200k
              development sample with the same 5x3 repeated folds used for the
              finalists, and paired fold-by-fold against logistic regression.
  3. TEST     Refit on all 800k training rows (three seeds) and score the 200k
              held-out test set once. The test set plays no part in stages 1-2.

Outputs: results_v2_dl_tuning_*.csv in this folder.
"""
import json
import os
import sys
import time

SMOKE = "--smoke" in sys.argv
os.environ["MPLBACKEND"] = "Agg"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ---- Reuse the notebook's definitions --------------------------------------
NB = json.load(open("thesis_churn_v2.ipynb"))
G = {"__name__": "__main__"}


def run_cell(i):
    src = "".join(NB["cells"][i]["source"])
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    exec(compile(src, f"<notebook cell {i}>", "exec"), G)


for i in (3, 5):
    run_cell(i)
# The search is run on the CPU: it gives the same model as the GPU and the
# notebook's device benchmark measured it 5-8x faster for networks of this size.
G["DL_DEVICE"] = "CPU"
G["HARDWARE"] = {}
for i in (10, 11, 23, 54):
    run_cell(i)
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

G["PRUNED_FEATURES"] = list(pd.read_csv("results_v2_fe_prune_log.csv")["dropped"])
for i in (75, 79, 81, 83, 88, 102, 104, 106, 108, 109):
    run_cell(i)

keras, layers, tf = G["keras"], G["layers"], G["tf"]
KerasChurnClassifier, LeakageSafeClassifier = G["KerasChurnClassifier"], G["LeakageSafeClassifier"]
FEATURE_SETS, WINSORISE_COLS, RANDOM_STATE = G["FEATURE_SETS"], G["WINSORISE_COLS"], G["RANDOM_STATE"]
X_train_full, y_train_full, X_test, y_test = G["X_train_full"], G["y_train_full"], G["X_test"], G["y_test"]
X_dev, y_dev, X_tune, y_tune = G["X_dev"], G["y_dev"], G["X_tune"], G["y_tune"]

from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 4 if SMOKE else 50
MAX_EPOCHS = 4 if SMOKE else 60
PATIENCE = 2 if SMOKE else 8
if SMOKE:
    X_tune, y_tune = X_tune.iloc[:9000], y_tune.iloc[:9000]
    X_dev, y_dev = X_dev.iloc[:12000], y_dev.iloc[:12000]
    X_train_full, y_train_full = X_train_full.iloc[:20000], y_train_full.iloc[:20000]
    X_test, y_test = X_test.iloc[:10000], y_test.iloc[:10000]


def make_builder(p):
    """Network factory for one hyperparameter configuration."""

    def builder(input_dim):
        inp = layers.Input(shape=(input_dim,))
        x = inp
        reg = keras.regularizers.l2(p["l2"]) if p["l2"] > 0 else None
        width = p["width"]
        for k in range(p["n_layers"]):
            x = layers.Dense(width, kernel_regularizer=reg,
                             kernel_initializer="lecun_normal" if p["activation"] == "selu" else "glorot_uniform")(x)
            if p["batchnorm"]:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(p["activation"])(x)
            if p["dropout"] > 0:
                x = (layers.AlphaDropout if p["activation"] == "selu" else layers.Dropout)(p["dropout"])(x)
            if p["shape"] == "taper":
                width = max(8, width // 2)
        logit = layers.Dense(1)(x)
        if p["wide"]:
            # "Wide and deep": a linear path from the inputs straight to the logit,
            # so the network contains logistic regression as a special case.
            logit = layers.Add()([logit, layers.Dense(1, use_bias=False)(inp)])
        out = layers.Activation("sigmoid")(logit)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(p["lr"]), loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
        return model

    return builder


def fit_net(p, X, y, seed=RANDOM_STATE):
    spec = FEATURE_SETS[p["feature_set"]]
    net = KerasChurnClassifier(make_builder(p), spec["numeric"], spec["categorical"],
                               winsorise_cols=WINSORISE_COLS, imbalance="none",
                               epochs=MAX_EPOCHS, batch_size=p["batch_size"], patience=PATIENCE,
                               random_state=seed)
    t0 = time.perf_counter()
    net.fit(X[spec["features"]], y)
    return net, time.perf_counter() - t0


def predict(net, p, X):
    return net.predict_proba(X[FEATURE_SETS[p["feature_set"]]["features"]])[:, 1]


def suggest(trial):
    return dict(
        feature_set=trial.suggest_categorical("feature_set", ["baseline", "engineered"]),
        n_layers=trial.suggest_int("n_layers", 1, 5),
        width=trial.suggest_categorical("width", [16, 32, 64, 128, 256, 512]),
        shape=trial.suggest_categorical("shape", ["constant", "taper"]),
        activation=trial.suggest_categorical("activation", ["relu", "gelu", "silu", "selu", "tanh"]),
        batchnorm=trial.suggest_categorical("batchnorm", [False, True]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        l2=trial.suggest_float("l2", 1e-7, 1e-3, log=True),
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        wide=trial.suggest_categorical("wide", [False, True]),
    )


# ---- Stage 1: search --------------------------------------------------------
print(f"STAGE 1 — Optuna search: {N_TRIALS} trials, 3-fold CV on {len(y_tune):,} tuning rows", flush=True)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
tune_folds = list(skf.split(X_tune, y_tune))
trial_rows = []


def objective(trial):
    p = suggest(trial)
    scores, secs, epochs = [], 0.0, []
    for k, (tr, va) in enumerate(tune_folds):
        net, s = fit_net(p, X_tune.iloc[tr], y_tune.iloc[tr])
        scores.append(roc_auc_score(y_tune.iloc[va], predict(net, p, X_tune.iloc[va])))
        secs += s
        epochs.append(net.epochs_run_)
        keras.backend.clear_session()
        trial.report(float(np.mean(scores)), k)
        if k == 0 and trial.should_prune():
            trial_rows.append({"trial": trial.number, "cv_auc": scores[0], "folds": 1, "seconds": secs,
                               "state": "pruned", **p})
            raise optuna.TrialPruned()
    trial_rows.append({"trial": trial.number, "cv_auc": float(np.mean(scores)), "cv_auc_std": float(np.std(scores)),
                       "folds": 3, "seconds": secs, "mean_epochs": float(np.mean(epochs)), "state": "complete", **p})
    print(f"  trial {trial.number:>2d}  AUC {np.mean(scores):.4f}  {secs:5.0f}s  {p}", flush=True)
    pd.DataFrame(trial_rows).to_csv("results_v2_dl_tuning_trials.csv", index=False)
    return float(np.mean(scores))


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
# Seed the search with the study's two literature configurations, so the tuned
# network can never be worse on the search criterion than the defaults.
study.enqueue_trial(dict(feature_set="engineered", n_layers=2, width=64, shape="taper", activation="relu",
                         batchnorm=False, dropout=0.25, l2=1e-7, lr=1e-3, batch_size=1024, wide=False))
study.enqueue_trial(dict(feature_set="engineered", n_layers=5, width=256, shape="taper", activation="relu",
                         batchnorm=True, dropout=0.3, l2=1e-7, lr=1e-3, batch_size=1024, wide=False))
t_search = time.perf_counter()
study.optimize(objective, n_trials=N_TRIALS)
search_seconds = time.perf_counter() - t_search
pd.DataFrame(trial_rows).to_csv("results_v2_dl_tuning_trials.csv", index=False)
best = dict(study.best_params)
print(f"\nBest configuration (CV AUC {study.best_value:.4f}, search {search_seconds / 60:.0f} min):\n  {best}", flush=True)

# ---- Stage 2: confirm on the finalists' repeated folds ----------------------
print("\nSTAGE 2 — 5x3 repeated CV on the development sample, paired with logistic regression", flush=True)
splitter = RepeatedStratifiedKFold(n_splits=G["N_SPLITS"], n_repeats=1 if SMOKE else G["N_REPEATS"],
                                   random_state=RANDOM_STATE)
lr_spec = FEATURE_SETS["baseline"]
fold_rows = []
for fold, (tr, va) in enumerate(splitter.split(X_dev[FEATURE_SETS["engineered"]["features"]], y_dev)):
    lr = LeakageSafeClassifier(LogisticRegression(C=1.0, max_iter=1000), numeric_features=lr_spec["numeric"],
                               categorical_features=lr_spec["categorical"], winsorise_cols=WINSORISE_COLS,
                               imbalance="none", random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    lr.fit(X_dev[lr_spec["features"]].iloc[tr], y_dev.iloc[tr])
    lr_s = time.perf_counter() - t0
    auc_lr = roc_auc_score(y_dev.iloc[va], lr.predict_proba(X_dev[lr_spec["features"]].iloc[va])[:, 1])
    net, net_s = fit_net(best, X_dev.iloc[tr], y_dev.iloc[tr])
    auc_net = roc_auc_score(y_dev.iloc[va], predict(net, best, X_dev.iloc[va]))
    keras.backend.clear_session()
    fold_rows.append({"fold": fold, "auc_logistic": auc_lr, "auc_tuned_network": auc_net,
                      "train_seconds_logistic": lr_s, "train_seconds_tuned_network": net_s,
                      "epochs_run": net.epochs_run_})
    print(f"  fold {fold:>2d}  LR {auc_lr:.4f}  tuned net {auc_net:.4f}  diff {auc_net - auc_lr:+.4f}", flush=True)
    pd.DataFrame(fold_rows).to_csv("results_v2_dl_tuning_cv_folds.csv", index=False)
folds = pd.DataFrame(fold_rows)
d = folds.auc_tuned_network - folds.auc_logistic
rng = np.random.default_rng(RANDOM_STATE)
boot = [rng.choice(d, len(d)).mean() for _ in range(10_000)]

# ---- Stage 3: one look at the test set --------------------------------------
print("\nSTAGE 3 — refit on the full training partition, score the held-out test set", flush=True)
lr = LeakageSafeClassifier(LogisticRegression(C=1.0, max_iter=1000), numeric_features=lr_spec["numeric"],
                           categorical_features=lr_spec["categorical"], winsorise_cols=WINSORISE_COLS,
                           imbalance="none", random_state=RANDOM_STATE)
t0 = time.perf_counter()
lr.fit(X_train_full[lr_spec["features"]], y_train_full)
lr_fit_s = time.perf_counter() - t0
p_lr = lr.predict_proba(X_test[lr_spec["features"]])[:, 1]
test_rows = [{"model": "LogisticRegression [A]", "roc_auc": roc_auc_score(y_test, p_lr),
              "pr_auc": average_precision_score(y_test, p_lr), "train_seconds": lr_fit_s}]
seed_preds = []
one = X_test.iloc[[0]]
for seed in ([42] if SMOKE else [42, 43, 44]):
    net, s = fit_net(best, X_train_full, y_train_full, seed=seed)
    p = predict(net, best, X_test)
    seed_preds.append(p)
    lat = []
    for _ in range(12):
        t0 = time.perf_counter()
        predict(net, best, one)
        lat.append(1000 * (time.perf_counter() - t0))
    test_rows.append({"model": f"Tuned network (seed {seed})", "roc_auc": roc_auc_score(y_test, p),
                      "pr_auc": average_precision_score(y_test, p), "train_seconds": s,
                      "epochs_run": net.epochs_run_, "online_p50_ms": float(np.median(lat[2:])),
                      "parameters": int(net.model_.count_params())})
    print(f"  seed {seed}: test AUC {test_rows[-1]['roc_auc']:.5f}  ({s:.0f}s, {net.epochs_run_} epochs)", flush=True)
    keras.backend.clear_session()
p_ens = np.mean(seed_preds, axis=0)
test_rows.append({"model": "Tuned network (mean of seeds)", "roc_auc": roc_auc_score(y_test, p_ens),
                  "pr_auc": average_precision_score(y_test, p_ens)})

# Paired bootstrap of the test-set AUC difference (seed-42 network minus logistic regression).
yt = np.asarray(y_test)
diffs = []
for _ in range(100 if SMOKE else 1000):
    idx = rng.integers(0, len(yt), len(yt))
    diffs.append(roc_auc_score(yt[idx], seed_preds[0][idx]) - roc_auc_score(yt[idx], p_lr[idx]))
# Capture at the campaign depth used in the paper (top 2 %).
k = int(0.02 * len(yt))
capture = {name: int(yt[np.argsort(-p)[:k]].sum()) for name, p in
           [("logistic", p_lr), ("tuned_network", seed_preds[0]), ("tuned_network_ensemble", p_ens)]}

test = pd.DataFrame(test_rows)
test.to_csv("results_v2_dl_tuning_test.csv", index=False)
summary = pd.DataFrame([
    ("search_trials", N_TRIALS), ("search_minutes", search_seconds / 60),
    ("best_search_cv_auc", study.best_value), ("best_params", json.dumps(best)),
    ("cv_auc_logistic", folds.auc_logistic.mean()), ("cv_auc_tuned_network", folds.auc_tuned_network.mean()),
    ("cv_mean_diff_network_minus_logistic", d.mean()),
    ("cv_diff_ci95_low", float(np.percentile(boot, 2.5))), ("cv_diff_ci95_high", float(np.percentile(boot, 97.5))),
    ("cv_paired_t_p", float(stats.ttest_rel(folds.auc_tuned_network, folds.auc_logistic).pvalue)),
    ("cv_folds_network_ahead", int((d > 0).sum())), ("cv_n_folds", len(d)),
    ("test_auc_logistic", test.roc_auc.iloc[0]), ("test_auc_tuned_network_seed42", test.roc_auc.iloc[1]),
    ("test_auc_tuned_network_ensemble", test.roc_auc.iloc[-1]),
    ("test_diff_ci95_low", float(np.percentile(diffs, 2.5))), ("test_diff_ci95_high", float(np.percentile(diffs, 97.5))),
    ("test_capture_top2pct", json.dumps(capture)), ("materiality_threshold", 0.005), ("smoke", SMOKE),
], columns=["item", "value"])
summary.to_csv("results_v2_dl_tuning_summary.csv", index=False)
print("\n" + summary.to_string(index=False))
print("\n" + test.round(5).to_string(index=False))
