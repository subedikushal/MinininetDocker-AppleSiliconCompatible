"""
retrain_mininet.py — Mininet-Calibrated IDS Model Training
CS 8027: Advanced Networking Architecture

WHY THIS SCRIPT EXISTS
══════════════════════
The InSDN dataset was captured on real SDN hardware running real Metasploit
attacks.  Mininet is a software emulator that rate-limits all traffic through
the host kernel's network stack.

Concrete numbers from this project:
  InSDN DDoS flow  : ~10,000,000 pkts/s, ~400 MB/s byte rate
  Mininet ping -f  :         ~71 pkts/s, ~0.2 MB/s byte rate  ← 140,000× slower
  Mininet hping3   :    ~10,000 pkts/s, ~0.6 MB/s byte rate   ←    1,000× slower

A model trained on InSDN data has never seen any flow with 71 pps labeled
"Attack".  It classifies everything in Mininet as Normal with prob ≈ 0.

SOLUTION
════════
Train a fresh set of tree-based models on SYNTHETIC flow samples whose
statistical distributions are calibrated to what Mininet actually produces.
The synthetic data is derived from the observed controller logs throughout
this project.

The new models are saved alongside the InSDN models so the controller can
be switched between them by editing model_info.json.

OUTPUT FILES  (written to MODEL_DIR, default /app/projects)
════════════
  model_dt_mininet.joblib    Decision Tree
  model_rf_mininet.joblib    Random Forest
  model_info.json            Updated manifest — points to best Mininet model

USAGE
═════
  python3 /app/projects/retrain_mininet.py
  # then restart:  ryu-manager projects/controller.py

NO GOOGLE COLAB NEEDED — runs entirely on the Docker host.
Dependencies: scikit-learn, numpy, joblib  (already installed in the container)
"""

import json
import os
import time

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/projects")
SEED      = 42
rng       = np.random.default_rng(SEED)

FEATURE_NAMES = [
    "Flow Duration (s)",   # [0]
    "Tot Fwd Pkts",        # [1]
    "Tot Bwd Pkts",        # [2]
    "Flow Byts/s",         # [3]
    "Flow Pkts/s",         # [4]
    "Subflow Fwd Pkts",    # [5] ≡ [1]
    "Subflow Bwd Pkts",    # [6] ≡ [2]
    "Subflow Fwd Byts",    # [7]
    "Subflow Bwd Byts",    # [8]
]

SEP = "═" * 68


def build_feature_row(dur_s, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes):
    """Construct one feature vector matching the controller's output exactly."""
    dur_s = max(dur_s, 1e-9)
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts  = fwd_pkts  + bwd_pkts
    return [
        dur_s,
        fwd_pkts,
        bwd_pkts,
        total_bytes / dur_s,    # Flow Byts/s
        total_pkts  / dur_s,    # Flow Pkts/s
        fwd_pkts,               # Subflow Fwd Pkts ≡ Tot Fwd Pkts
        bwd_pkts,               # Subflow Bwd Pkts ≡ Tot Bwd Pkts
        fwd_bytes,              # Subflow Fwd Byts
        bwd_bytes,              # Subflow Bwd Byts
    ]


# ── Normal traffic generators ─────────────────────────────────────────────────

def gen_normal_pingall(n):
    """
    ARP / ICMP echo from pingall and normal host communication.
    Observed: pkts=5, dur=5-15s, byts_s=35-70
    """
    dur   = rng.uniform(0.5, 20.0, n)
    pkts  = rng.integers(1, 30, n).astype(float)
    sym   = rng.uniform(0.7, 1.3, n)
    bpkts = np.clip(pkts * sym, 1, pkts * 2)
    psize = rng.uniform(64, 200, n)
    fb    = pkts  * psize
    bb    = bpkts * psize
    rows  = [build_feature_row(d, fp, bp, f, b)
             for d, fp, bp, f, b in zip(dur, pkts, bpkts, fb, bb)]
    return np.array(rows)


def gen_normal_iperf_tcp(n):
    """
    Legitimate iperf3 TCP transfers (h1/h2/h3 → web_srv/db_srv).
    Observed: pkts grows to 160K+, byts_s up to 327 MB/s, duration 0.5-30s
    """
    dur    = rng.uniform(0.5, 60.0, n)
    byts_s = rng.uniform(1e6, 350e6, n)          # 1 MB/s – 350 MB/s
    total_bytes = byts_s * dur
    psize  = rng.uniform(1200, 1500, n)           # near-MTU TCP segments
    # TCP: roughly 60% payload forward, 40% ACKs backward
    fwd_bytes = total_bytes * rng.uniform(0.55, 0.65, n)
    bwd_bytes = total_bytes - fwd_bytes
    fwd_pkts  = fwd_bytes / psize
    bwd_pkts  = bwd_bytes / 60                    # ACKs are small (40 bytes typ.)
    rows = [build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)]
    return np.array(rows)


def gen_normal_web(n):
    """
    Short HTTP-like request/response flows.
    Moderate byte rates, short durations, asymmetric (server sends more data).
    """
    dur       = rng.uniform(0.05, 10.0, n)
    req_bytes = rng.uniform(200, 2000, n)          # small client request
    resp_ratio = rng.uniform(1, 100, n)            # server response 1-100× request
    resp_bytes = req_bytes * resp_ratio
    fwd_pkts   = np.ceil(req_bytes  / 1400)
    bwd_pkts   = np.ceil(resp_bytes / 1400)
    rows = [build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in
            zip(dur, fwd_pkts, bwd_pkts, req_bytes, resp_bytes)]
    return np.array(rows)


# ── Attack traffic generators ─────────────────────────────────────────────────

def gen_attack_icmp_flood(n):
    """
    ping -f -s PAYLOAD  in Mininet.
    Observed: 71 pps over 122s with 1428-byte packets.
    Range: 50–300 pps (Mininet kernel-limited), 10–300 s duration.
    Distinguishing signature: sustained moderate rate, nearly symmetric,
    long duration, small-to-mid byte rate (100 KB/s – 3 MB/s).
    """
    pps      = rng.uniform(50, 300, n)
    dur      = rng.uniform(10, 300, n)
    psize    = rng.uniform(100, 1500, n)           # -s flag controls this
    fwd_pkts = pps * dur
    # Mininet ICMP: reply rate ≈ 95-100% (near-zero loss in emulation)
    loss     = rng.uniform(0.0, 0.05, n)
    bwd_pkts = fwd_pkts * (1 - loss)
    fwd_bytes = fwd_pkts * psize
    bwd_bytes = bwd_pkts * psize
    rows = [build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)]
    return np.array(rows)


def gen_attack_syn_flood(n):
    """
    hping3 -S --flood -p 80 TARGET  in Mininet.
    Generates TCP SYN packets as fast as the kernel allows.
    Mininet can produce 10 K – 1 M pps depending on host CPU.
    Distinguishing signature: very high fwd pkt rate, small packets (60 bytes),
    bwd_pkts << fwd_pkts (server SYN-ACKs may not all return under flood).
    """
    pps      = rng.uniform(5000, 500000, n)
    dur      = rng.uniform(2, 60, n)
    fwd_pkts = pps * dur
    # SYN-ACK return ratio: 5-50% depending on server backlog exhaustion
    bwd_ratio = rng.uniform(0.05, 0.50, n)
    bwd_pkts  = fwd_pkts * bwd_ratio
    fwd_bytes = fwd_pkts * 60    # TCP SYN: 60 bytes
    bwd_bytes = bwd_pkts * 60    # TCP SYN-ACK: 60 bytes
    rows = [build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)]
    return np.array(rows)


def gen_attack_udp_flood_with_server(n):
    """
    h_attack iperf -c TARGET -u -b 1000M -t T  WITH a UDP iperf server running.
    The server sends back UDP stats every second → bwd_bytes is non-trivial.
    Distinguishing signature: extremely high byts_s, high fwd pkts,
    lower-than-TCP bwd ratio (UDP not ACK-based).
    Observed: pkts=2.6M in 30s, byts_s=100-400 MB/s
    """
    pps      = rng.uniform(10000, 2000000, n)
    dur      = rng.uniform(5, 60, n)
    fwd_pkts = pps * dur
    fwd_bytes = fwd_pkts * 1470                    # iperf UDP default datagram
    # UDP iperf server sends periodic stats: ~1 pkt/s at ~200 bytes
    bwd_pkts  = dur * rng.uniform(0.5, 2.0, n)
    bwd_bytes = bwd_pkts * 200
    rows = [build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)]
    return np.array(rows)


def gen_attack_slow_scan(n):
    """
    Slow probe / port scan from attack host.
    Many short flows to different targets — here represented as one long
    aggregate flow with moderate packet rate and tiny byte count.
    """
    dur      = rng.uniform(30, 300, n)
    pps      = rng.uniform(5, 50, n)
    fwd_pkts = pps * dur
    bwd_pkts = fwd_pkts * rng.uniform(0.0, 0.3, n)  # few/no responses
    psize    = rng.uniform(40, 100, n)
    fwd_bytes = fwd_pkts * psize
    bwd_bytes = bwd_pkts * psize
    rows = [build_feature_row(d, fp, bp, f, b)
            for d, fp, bp, f, b in zip(dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes)]
    return np.array(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print(" Mininet-Calibrated IDS Model Training")
    print(SEP)
    print(f" Output directory : {MODEL_DIR}")
    print(f" Random seed      : {SEED}\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Generate synthetic dataset ────────────────────────────────────────────
    print("Generating synthetic flow samples...")

    N_PER_GROUP = 8000

    X_normal = np.vstack([
        gen_normal_pingall(N_PER_GROUP),
        gen_normal_iperf_tcp(N_PER_GROUP),
        gen_normal_web(N_PER_GROUP),
    ])
    y_normal = np.zeros(len(X_normal), dtype=int)

    X_attack = np.vstack([
        gen_attack_icmp_flood(N_PER_GROUP),
        gen_attack_syn_flood(N_PER_GROUP),
        gen_attack_udp_flood_with_server(N_PER_GROUP),
        gen_attack_slow_scan(N_PER_GROUP),
    ])
    y_attack = np.ones(len(X_attack), dtype=int)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    # Replace inf/nan that can arise from division by near-zero duration
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=0.0)

    print(f"  Normal  samples : {len(X_normal):,}")
    print(f"  Attack  samples : {len(X_attack):,}")
    print(f"  Total           : {len(X):,}")

    # ── Verify that the observed ICMP flood IS in the attack region ───────────
    # From user's actual ping -f -s 1400 run: 8727 pkts / 122.854 s
    dur_s     = 122.854
    fwd_pkts  = 8727
    bwd_pkts  = 8724
    pkt_size  = 1428
    fwd_bytes = fwd_pkts * pkt_size
    bwd_bytes = bwd_pkts * pkt_size
    observed_icmp = np.array([build_feature_row(
        dur_s, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes
    )])
    print(f"\n  Observed ICMP flood feature vector:")
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, observed_icmp[0])):
        print(f"    [{i}] {name:<25} = {val:.4g}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── Train Decision Tree ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(" Training Decision Tree (Mininet-calibrated)")
    print(SEP)

    t0 = time.perf_counter()
    dt = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=SEED,
    )
    dt.fit(X_train, y_train)
    dt_time = time.perf_counter() - t0

    y_pred_dt  = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)[:, 1]
    dt_f1  = f1_score(y_test, y_pred_dt)
    dt_auc = roc_auc_score(y_test, y_proba_dt)
    dt_acc = accuracy_score(y_test, y_pred_dt)

    print(f"  Accuracy  : {dt_acc:.4f}")
    print(f"  F1        : {dt_f1:.4f}")
    print(f"  AUC       : {dt_auc:.4f}")
    print(f"  Train time: {dt_time:.1f}s")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_dt,
                                 target_names=["Normal", "Attack"]))

    # Print top levels of the learned tree (human-readable rules)
    print("  Learned decision rules (first 4 levels):")
    rules = export_text(dt, feature_names=FEATURE_NAMES, max_depth=4, decimals=2)
    for line in rules.splitlines()[:40]:
        print("    " + line)
    print("    ...")

    # ── Train Random Forest ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(" Training Random Forest (Mininet-calibrated)")
    print(SEP)

    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0

    y_pred_rf  = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    rf_f1  = f1_score(y_test, y_pred_rf)
    rf_auc = roc_auc_score(y_test, y_proba_rf)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    print(f"  Accuracy  : {rf_acc:.4f}")
    print(f"  F1        : {rf_f1:.4f}")
    print(f"  AUC       : {rf_auc:.4f}")
    print(f"  Train time: {rf_time:.1f}s")

    # ── Self-test: classify the observed ICMP flood ───────────────────────────
    print(f"\n{SEP}")
    print(" Self-test: classify observed ICMP flood (8727 pkts / 122s)")
    print(SEP)

    prob_dt = dt.predict_proba(observed_icmp)[0][1]
    prob_rf = rf.predict_proba(observed_icmp)[0][1]
    print(f"  Decision Tree : prob = {prob_dt:.4f}  → "
          f"{'ATTACK ✅' if prob_dt >= 0.45 else 'Normal ❌'}")
    print(f"  Random Forest : prob = {prob_rf:.4f}  → "
          f"{'ATTACK ✅' if prob_rf >= 0.45 else 'Normal ❌'}")

    # Additional test vectors
    print(f"\n  Additional test vectors:")
    tests = [
        ("Normal pingall (5 pkts)", 10.0, 3, 2, 3*100, 2*100),
        ("Normal iperf (h1, 30s)", 30.0, 160000, 80000, 160000*1450, 80000*60),
        ("ICMP flood (71 pps, 122s)", 122.0, 8727, 8724, 8727*1428, 8724*1428),
        ("Larger ICMP flood (200 pps, 60s)", 60.0, 12000, 11970, 12000*1428, 11970*1428),
        ("hping3 SYN (50K pps, 30s)", 30.0, 1500000, 200000, 1500000*60, 200000*60),
        ("UDP iperf attack (30s)", 30.0, 2600000, 30, 2600000*1470, 30*200),
    ]

    print(f"  {'Scenario':<42} {'DT prob':>8}  {'RF prob':>8}  Result")
    print("  " + "-" * 75)
    for name, dur, fp, bp, fb, bb in tests:
        vec = np.array([build_feature_row(dur, fp, bp, fb, bb)])
        p_dt = dt.predict_proba(vec)[0][1]
        p_rf = rf.predict_proba(vec)[0][1]
        res  = "ATTACK" if p_dt >= 0.45 else "Normal"
        mark = " ✅" if ("flood" in name.lower() or "syn" in name.lower() or "udp" in name.lower()) and res == "ATTACK" else ""
        mark = mark or (" ✅" if ("normal" in name.lower() or "iperf (h1" in name.lower()) and res == "Normal" else "")
        print(f"  {name:<42} {p_dt:>8.4f}  {p_rf:>8.4f}  {res}{mark}")

    # ── Save models ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(" Saving artefacts")
    print(SEP)

    dt_path = os.path.join(MODEL_DIR, "model_dt_mininet.joblib")
    rf_path = os.path.join(MODEL_DIR, "model_rf_mininet.joblib")

    joblib.dump(dt, dt_path)
    joblib.dump(rf, rf_path)
    print(f"  Saved: {dt_path}")
    print(f"  Saved: {rf_path}")

    # Choose best model by F1
    if dt_f1 >= rf_f1:
        winner_name = "Decision Tree (Mininet)"
        winner_file = "model_dt_mininet.joblib"
        winner_f1   = dt_f1
        winner_auc  = dt_auc
        winner_acc  = dt_acc
    else:
        winner_name = "Random Forest (Mininet)"
        winner_file = "model_rf_mininet.joblib"
        winner_f1   = rf_f1
        winner_auc  = rf_auc
        winner_acc  = rf_acc

    # ── Update model_info.json ────────────────────────────────────────────────
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    try:
        with open(info_path) as fp:
            info = json.load(fp)
    except FileNotFoundError:
        # Build from scratch if the InSDN notebook hasn't been run yet
        info = {
            "feature_names": FEATURE_NAMES,
            "training": {},
            "all_models": {},
        }

    info["winner"]              = winner_name
    info["winner_file"]         = winner_file
    info["decision_threshold"]  = 0.45
    info["note"] = (
        "Mininet-calibrated model trained on synthetic flow samples. "
        "Uses raw (unscaled) features in seconds/counts/bytes — "
        "same as the controller computes directly from OpenFlow stats."
    )
    info["mininet_models"] = {
        "Decision Tree (Mininet)": {
            "file":     "model_dt_mininet.joblib",
            "F1":       round(dt_f1, 4),
            "Accuracy": round(dt_acc, 4),
            "AUC":      round(dt_auc, 4),
        },
        "Random Forest (Mininet)": {
            "file":     "model_rf_mininet.joblib",
            "F1":       round(rf_f1, 4),
            "Accuracy": round(rf_acc, 4),
            "AUC":      round(rf_auc, 4),
        },
    }

    with open(info_path, "w") as fp:
        json.dump(info, fp, indent=2)
    print(f"  Updated: {info_path}")

    print(f"\n{SEP}")
    print(f" DONE — winner: {winner_name}")
    print(f"   F1       = {winner_f1:.4f}")
    print(f"   Accuracy = {winner_acc*100:.2f}%")
    print(f"   AUC      = {winner_auc:.4f}")
    print(SEP)
    print("""
 NEXT STEPS
 ──────────
 1. Restart the controller:
      ryu-manager projects/controller.py

 2. In Mininet, run the experiment:
      mininet> pingall
      mininet> web_srv iperf -s &
      mininet> h1 iperf -c 10.0.0.10 -t 30 &
      mininet> h_attack ping -f -s 1400 10.0.0.10

 3. Watch the controller log for:
      [CLASSIFY] ... bwd_bytes=XXXX  prob=0.9XX
      ATTACK DETECTED  00:00:00:00:00:20→00:00:00:00:00:10

 4. Verify isolation:
      mininet> h_attack ping 10.0.0.10 -c 5   # should be 100% packet loss
      mininet> h1 ping 10.0.0.10 -c 5          # should still pass
""")


if __name__ == "__main__":
    main()