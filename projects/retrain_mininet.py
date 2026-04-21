"""
retrain_mininet.py -- Mininet-Calibrated IDS Model Training (All 4 Models)
CS 8027: Advanced Networking Architecture

Updated to mirror InSDN Dataset distributions:
~20% Normal Traffic (ping, iperf, HTTP)
~80% Attack Traffic (DoS, DDoS, Probe, Brute Force, Botnet, Web/Exploit)
"""

import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed -- XGBoost will be skipped.")

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed -- LightGBM will be skipped.")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/projects/models/")
SEED = 42
rng = np.random.default_rng(SEED)
THRESHOLD = 0.45
ALPHA_F1 = 0.70
ALPHA_LAT = 0.30

FEATURE_NAMES = [
    "Flow Duration (s)",  # [0]
    "Tot Fwd Pkts",  # [1]
    "Tot Bwd Pkts",  # [2]
    "Flow Byts/s",  # [3]
    "Flow Pkts/s",  # [4]
    "Subflow Fwd Pkts",  # [5] identical to [1]
    "Subflow Bwd Pkts",  # [6] identical to [2]
    "Subflow Fwd Byts",  # [7]
    "Subflow Bwd Byts",  # [8]
]

SEP = "=" * 68

# ── Feature builder ───────────────────────────────────────────────────────────


def build_feature_row(dur_s, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes):
    """Return one 9-element feature vector matching the controller exactly."""
    dur_s = max(dur_s, 1e-9)
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = fwd_pkts + bwd_pkts
    return [
        dur_s,
        fwd_pkts,
        bwd_pkts,
        total_bytes / dur_s,
        total_pkts / dur_s,
        fwd_pkts,
        bwd_pkts,
        fwd_bytes,
        bwd_bytes,
    ]


# ── Normal traffic generators ─────────────────────────────────────────────────


def gen_normal_pingall(n):
    dur = rng.uniform(0.5, 20.0, n)
    pkts = rng.integers(1, 30, n).astype(float)
    bpkts = np.clip(pkts * rng.uniform(0.7, 1.3, n), 1, pkts * 2)
    psize = rng.uniform(64, 200, n)
    return np.array(
        [
            build_feature_row(d, fp, bp, fp * ps, bp * ps)
            for d, fp, bp, ps in zip(dur, pkts, bpkts, psize)
        ]
    )


def gen_normal_iperf_tcp(n):
    dur = rng.uniform(0.5, 60.0, n)
    byts_s = rng.uniform(1e6, 350e6, n)
    total_b = byts_s * dur
    psize = rng.uniform(1200, 1500, n)
    fwd_bytes = total_b * rng.uniform(0.55, 0.65, n)
    bwd_bytes = total_b - fwd_bytes
    fwd_pkts = fwd_bytes / psize
    bwd_pkts = bwd_bytes / 60
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_normal_web(n):
    dur = rng.uniform(0.05, 10.0, n)
    req_bytes = rng.uniform(200, 2000, n)
    resp_bytes = req_bytes * rng.uniform(1, 100, n)
    fwd_pkts = np.ceil(req_bytes / 1400)
    bwd_pkts = np.ceil(resp_bytes / 1400)
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, req_bytes, resp_bytes)
        ]
    )


# ── Attack traffic generators ─────────────────────────────────────────────────


def gen_attack_icmp_flood(n):
    pps = rng.uniform(50, 300, n)
    dur = rng.uniform(10, 300, n)
    psize = rng.uniform(100, 1500, n)
    fwd_pkts = pps * dur
    bwd_pkts = fwd_pkts * (1 - rng.uniform(0.0, 0.05, n))
    fwd_bytes = fwd_pkts * psize
    bwd_bytes = bwd_pkts * psize
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_attack_syn_flood(n):
    pps = rng.uniform(5000, 500000, n)
    dur = rng.uniform(2, 60, n)
    fwd_pkts = pps * dur
    bwd_pkts = fwd_pkts * rng.uniform(0.05, 0.50, n)
    fwd_bytes = fwd_pkts * 60
    bwd_bytes = bwd_pkts * 60
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_attack_udp_flood_with_server(n):
    pps = rng.uniform(10000, 2000000, n)
    dur = rng.uniform(5, 60, n)
    fwd_pkts = pps * dur
    fwd_bytes = fwd_pkts * 1470
    bwd_pkts = dur * rng.uniform(0.5, 2.0, n)
    bwd_bytes = bwd_pkts * 200
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_attack_slow_scan(n):
    dur = rng.uniform(30, 300, n)
    pps = rng.uniform(5, 50, n)
    fwd_pkts = pps * dur
    bwd_pkts = fwd_pkts * rng.uniform(0.0, 0.3, n)
    psize = rng.uniform(40, 100, n)
    fwd_bytes = fwd_pkts * psize
    bwd_bytes = bwd_pkts * psize
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_attack_brute_force(n):
    """Many short, repetitive failed login flows (SSH/FTP)"""
    dur = rng.uniform(0.1, 2.0, n)
    fwd_pkts = rng.uniform(10, 25, n)
    bwd_pkts = fwd_pkts * rng.uniform(0.8, 1.2, n)
    fwd_bytes = fwd_pkts * rng.uniform(40, 80, n)
    bwd_bytes = bwd_pkts * rng.uniform(60, 100, n)
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_attack_botnet(n):
    """Long duration, low packet C&C beacons"""
    dur = rng.uniform(60, 300, n)
    fwd_pkts = rng.uniform(5, 50, n)
    bwd_pkts = rng.uniform(5, 50, n)
    fwd_bytes = fwd_pkts * rng.uniform(40, 60, n)
    bwd_bytes = bwd_pkts * rng.uniform(40, 60, n)
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


def gen_attack_web_exploit(n):
    """Quick, slightly larger payload requests with minimal response (R2L, XSS)"""
    dur = rng.uniform(0.05, 1.0, n)
    fwd_pkts = rng.uniform(5, 20, n)
    bwd_pkts = rng.uniform(2, 10, n)
    fwd_bytes = fwd_pkts * rng.uniform(200, 800, n)  # Payload
    bwd_bytes = bwd_pkts * rng.uniform(40, 100, n)
    return np.array(
        [
            build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)
        ]
    )


# ── Evaluation helper ─────────────────────────────────────────────────────────


def evaluate_model(model, X_test, y_test, X_latency, n_warmup=5, n_repeats=30):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    for _ in range(n_warmup):
        model.predict(X_latency)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(X_latency)
        times.append(time.perf_counter() - t0)
    lat_us = (np.median(times) / len(X_latency)) * 1e6

    return f1, auc, acc, lat_us


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print(f"\n{SEP}")
    print(" Mininet-Calibrated IDS Model Training (InSDN Distribution)")
    print(SEP)
    print(f" Output : {MODEL_DIR}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Generate synthetic dataset ─────────────────────────────────────────
    print(f"\nGenerating synthetic flow samples (~20% Normal / ~80% Attack)...")

    # 20,000 Normal Records
    X_normal = np.vstack(
        [
            gen_normal_pingall(6000),
            gen_normal_iperf_tcp(6000),
            gen_normal_web(8000),
        ]
    )

    # 80,000 Attack Records (Weighted towards DoS/DDoS and Probe like InSDN)
    X_attack = np.vstack(
        [
            gen_attack_icmp_flood(16000),
            gen_attack_syn_flood(16000),
            gen_attack_udp_flood_with_server(15000),
            gen_attack_slow_scan(30000),  # Probe
            gen_attack_brute_force(1500),  # Brute Force
            gen_attack_web_exploit(1000),  # Web / Exploit
            gen_attack_botnet(500),  # Botnet
        ]
    )

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate(
        [np.zeros(len(X_normal), dtype=int), np.ones(len(X_attack), dtype=int)]
    )
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=0.0)

    print(f"  Normal : {len(X_normal):,}  (pingall, iperf-TCP, HTTP)")
    print(
        f"  Attack : {len(X_attack):,}  (DDoS, DoS, Probe, BruteForce, Web/Exploit, Botnet)"
    )
    print(f"  Total  : {len(X):,}")

    # ── 2. Split and Save to CSV ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    print(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    print(f"  Saving train/test datasets to CSV...")

    header_str = ",".join(FEATURE_NAMES + ["Label"])

    # Define formats: 9 float columns (%.6f), 1 integer column (%d)
    column_formats = ['%.6f'] * 9 + ['%d']

    train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
    np.savetxt(os.path.join(MODEL_DIR, "train_data.csv"), train_data, delimiter=",", header=header_str, comments="", fmt=column_formats)
    
    test_data = np.hstack((X_test, y_test.reshape(-1, 1)))
    np.savetxt(os.path.join(MODEL_DIR, "test_data.csv"), test_data, delimiter=",", header=header_str, comments="", fmt=column_formats)

    print(f"  Saved -> {os.path.join(MODEL_DIR, 'train_data.csv')}")
    print(f"  Saved -> {os.path.join(MODEL_DIR, 'test_data.csv')}")

    lat_idx = np.random.default_rng(SEED).choice(
        len(X_test), size=min(500, len(X_test)), replace=False
    )
    X_latency = X_test[lat_idx]

    # ── 3. Model definitions ──────────────────────────────────────────────────
    model_defs = [
        (
            "Decision Tree",
            "model_dt_mininet.joblib",
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED,
            ),
            True,
        ),
        (
            "Random Forest",
            "model_rf_mininet.joblib",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            ),
            True,
        ),
        (
            "XGBoost",
            "model_xgb_mininet.joblib",
            (
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    verbosity=0,
                    random_state=SEED,
                )
                if HAS_XGB
                else None
            ),
            HAS_XGB,
        ),
        (
            "LightGBM",
            "model_lgbm_mininet.joblib",
            (
                LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    num_leaves=31,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    verbose=-1,
                    random_state=SEED,
                )
                if HAS_LGBM
                else None
            ),
            HAS_LGBM,
        ),
    ]

    # ── 4. Train + evaluate + save ────────────────────────────────────────────
    results, trained = {}, {}

    for name, filename, clf, available in model_defs:
        if not available:
            continue

        print(f"\n{SEP}\n Training: {name}\n{SEP}")
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        f1, auc, acc, lat_us = evaluate_model(clf, X_test, y_test, X_latency)

        print(
            f"  Accuracy   : {acc:.4f}\n  F1         : {f1:.4f}\n  AUC        : {auc:.4f}\n  Latency    : {lat_us:.4f} us/flow"
        )
        print(f"  Train time : {elapsed:.1f} s")

        out_path = os.path.join(MODEL_DIR, filename)
        joblib.dump(clf, out_path)

        results[name] = {
            "file": filename,
            "F1": round(f1, 4),
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "lat_us": round(lat_us, 4),
        }
        trained[name] = clf

    # ── 5. Composite score and winner selection ────────────────────────────────
    print(
        f"\n{SEP}\n Composite Score  (alpha={ALPHA_F1}*F1 + {ALPHA_LAT}*speed_normalised)\n{SEP}"
    )
    min_lat = min(v["lat_us"] for v in results.values())
    for name, v in results.items():
        sn = min_lat / v["lat_us"]
        cs = round(ALPHA_F1 * v["F1"] + ALPHA_LAT * sn, 4)
        results[name]["speed_norm"] = round(sn, 4)
        results[name]["composite"] = cs

    print(
        f"  {'Model':<30}  {'F1':>8}  {'Lat(us)':>10}  {'SpeedNorm':>10}  {'Composite':>10}"
    )
    print("  " + "-" * 72)
    for name, v in results.items():
        print(
            f"  {name:<30}  {v['F1']:>8.4f}  {v['lat_us']:>10.4f}  {v['speed_norm']:>10.4f}  {v['composite']:>10.4f}"
        )

    winner_name = max(results, key=lambda n: results[n]["composite"])
    winner = results[winner_name]

    # ── 6. Update model_info.json ──────────────────────────────────────────────
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    try:
        with open(info_path) as fp:
            info = json.load(fp)
    except FileNotFoundError:
        info = {"feature_names": FEATURE_NAMES, "training": {}, "all_models": {}}

    info["winner"] = winner_name
    info["winner_file"] = winner["file"]
    info["decision_threshold"] = THRESHOLD
    info["mininet_models"] = {
        name: {k: v for k, v in data.items() if k != "speed_norm"}
        for name, data in results.items()
    }

    with open(info_path, "w") as fp:
        json.dump(info, fp, indent=2)

    print(f"\n{SEP}\n DONE \n{SEP}")
    print(
        f"  Winner    : {winner_name}\n  File      : {winner['file']}\n  Composite : {winner['composite']:.4f}"
    )


if __name__ == "__main__":
    main()
