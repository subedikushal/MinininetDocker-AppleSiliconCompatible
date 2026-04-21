import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from threading import Thread, Event

# Guard: must be root
if os.geteuid() != 0:
    sys.exit(
        "Must run as root (Mininet requires it).\n"
        "  sudo python3 /app/projects/experiment.py"
    )

from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.clean import cleanup as mn_cleanup

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/projects/models")
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/app/projects")
RESULTS_BASE_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 6653
RYU_MANAGER = shutil.which("ryu-manager") or "/usr/local/bin/ryu-manager"
CONTROLLER_SCRIPT = os.path.join(PROJECT_DIR, "controller.py")

PHASE_NORMAL_S = 30
PHASE_ATTACK_S = 60
PHASE_RECOVERY_S = 30
LINK_DELAY = "2ms"

MODEL_CONFIGS = {
    "insdn": {
        "label": "InSDN Model (Real-Hardware Dataset)",
        "winner": "Decision Tree",
        "winner_file": "model_dt.joblib",
    },
    "mininet": {
        "label": "Mininet Model (Emulation-Calibrated)",
        "winner": "Decision Tree(Mininet)",
        "winner_file": "model_dt_mininet.joblib",
    },
}

# Updated MAC Names for Multi-Tier Topology
MAC_NAMES = {
    "00:00:00:00:00:01": "h1",
    "00:00:00:00:00:02": "h2",
    "00:00:00:00:00:03": "h3",
    "00:00:00:00:00:04": "h4",
    "00:00:00:00:00:05": "h5",
    "00:00:00:00:00:10": "web_srv",
    "00:00:00:00:00:11": "db_srv",
    "00:00:00:00:00:12": "api_srv",
    "00:00:00:00:00:15": "dns_srv",
    "00:00:00:00:00:20": "h_attack",
    "00:00:00:00:00:21": "h_attack2",
}

RE_CLASSIFY = re.compile(
    r"\[CLASSIFY\]\s+"
    r"(?P<src>[0-9a-f:]{17})->(?P<dst>[0-9a-f:]{17})"
    r"\s+pkts=(?P<pkts>\d+)"
    r"\s+dur_s=(?P<dur_s>[0-9.]+)"
    r"\s+byts_s=(?P<byts_s>[0-9.]+)"
    r"\s+bwd_bytes=(?P<bwd_bytes>\d+)"
    r"\s+prob=(?P<prob>[0-9.]+)"
)

RE_ATTACK = re.compile(
    r"ATTACK DETECTED\s+"
    r"(?P<src>[0-9a-f:]{17})->(?P<dst>[0-9a-f:]{17})"
    r"\s+prob=(?P<prob>[0-9.]+)"
)

RE_CLASSIFY_ARR = re.compile(
    r"\[CLASSIFY\]\s+"
    r"(?P<src>[0-9a-f:]{17})"
    r"[\xe2\x80\x94\xe2\x86\x92\u2192\u2014>]+"
    r"(?P<dst>[0-9a-f:]{17})"
    r"\s+pkts=(?P<pkts>\d+)"
    r"\s+dur_s=(?P<dur_s>[0-9.]+)"
    r"\s+byts_s=(?P<byts_s>[0-9.]+)"
    r"\s+bwd_bytes=(?P<bwd_bytes>\d+)"
    r"\s+prob=(?P<prob>[0-9.]+)"
)
RE_ATTACK_ARR = re.compile(
    r"ATTACK DETECTED\s+"
    r"(?P<src>[0-9a-f:]{17})"
    r"[\xe2\x80\x94\xe2\x86\x92\u2192\u2014>]+"
    r"(?P<dst>[0-9a-f:]{17})"
    r"\s+prob=(?P<prob>[0-9.]+)"
)


def _match_line(line):
    for pat in [RE_CLASSIFY, RE_CLASSIFY_ARR]:
        m = pat.search(line)
        if m:
            return "classify", m
    for pat in [RE_ATTACK, RE_ATTACK_ARR]:
        m = pat.search(line)
        if m:
            return "attack", m
    return None, None


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def patch_model_info(config_key):
    cfg = MODEL_CONFIGS[config_key]
    with open(MODEL_INFO_PATH) as fp:
        info = json.load(fp)
    info["winner"] = cfg["winner"]
    info["winner_file"] = cfg["winner_file"]
    with open(MODEL_INFO_PATH, "w") as fp:
        json.dump(info, fp, indent=2)


def wait_for_port(host, port, timeout=45):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"Controller not ready on {host}:{port} after {timeout}s")


class LogTailer:
    def __init__(self, log_path, experiment_start):
        self._path = log_path
        self._start = experiment_start
        self._events = []
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def events(self):
        return list(self._events)

    def _run(self):
        while not os.path.exists(self._path) and not self._stop.is_set():
            time.sleep(0.1)
        with open(self._path, "r", errors="replace") as fp:
            while not self._stop.is_set():
                line = fp.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                self._handle(line.rstrip(), time.time())

    def _handle(self, line, now):
        rel_s = round(now - self._start, 2)
        kind, m = _match_line(line)
        if m is None:
            return
        src, dst = m.group("src"), m.group("dst")
        if kind == "classify":
            self._events.append(
                {
                    "type": "classify",
                    "rel_s": rel_s,
                    "src": src,
                    "dst": dst,
                    "src_name": MAC_NAMES.get(src, src),
                    "dst_name": MAC_NAMES.get(dst, dst),
                    "pkts": int(m.group("pkts")),
                    "dur_s": float(m.group("dur_s")),
                    "byts_s": float(m.group("byts_s")),
                    "bwd_bytes": int(m.group("bwd_bytes")),
                    "prob": float(m.group("prob")),
                    "is_attack_flow": src == "00:00:00:00:00:20",
                }
            )
        else:
            self._events.append(
                {
                    "type": "detection",
                    "rel_s": rel_s,
                    "src": src,
                    "dst": dst,
                    "src_name": MAC_NAMES.get(src, src),
                    "dst_name": MAC_NAMES.get(dst, dst),
                    "prob": float(m.group("prob")),
                    "is_attack_flow": src == "00:00:00:00:00:20",
                }
            )


class RyuProcess:
    def __init__(self, log_path):
        self.log_path = log_path
        self.proc = None
        self._log_fp = None

    def start(self):
        log(f"  Starting ryu-manager -> {self.log_path}")
        self._log_fp = open(self.log_path, "w", buffering=1)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [RYU_MANAGER, CONTROLLER_SCRIPT],
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        wait_for_port(CONTROLLER_IP, CONTROLLER_PORT, timeout=45)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log_fp:
            self._log_fp.flush()
            self._log_fp.close()


# ── Multi-Tier Topology for Experiment ────────────────────────────────────────
def build_topology():
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )
    net.addController(
        "c0", controller=RemoteController, ip=CONTROLLER_IP, port=CONTROLLER_PORT
    )

    # Switches
    s_core = net.addSwitch("s100", protocols="OpenFlow13")
    s_agg1 = net.addSwitch("s200", protocols="OpenFlow13")
    s_agg2 = net.addSwitch("s201", protocols="OpenFlow13")
    s_user_a = net.addSwitch("s1", protocols="OpenFlow13")
    s_user_b = net.addSwitch("s2", protocols="OpenFlow13")
    s_server = net.addSwitch("s3", protocols="OpenFlow13")
    s_dmz = net.addSwitch("s4", protocols="OpenFlow13")

    # Hosts
    hosts = {
        "h1": net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01"),
        "h2": net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02"),
        "h3": net.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03"),
        "h4": net.addHost("h4", ip="10.0.0.4/24", mac="00:00:00:00:00:04"),
        "h5": net.addHost("h5", ip="10.0.0.5/24", mac="00:00:00:00:00:05"),
        "web_srv": net.addHost("web_srv", ip="10.0.0.10/24", mac="00:00:00:00:00:10"),
        "db_srv": net.addHost("db_srv", ip="10.0.0.11/24", mac="00:00:00:00:00:11"),
        "api_srv": net.addHost("api_srv", ip="10.0.0.12/24", mac="00:00:00:00:00:12"),
        "dns_srv": net.addHost("dns_srv", ip="10.0.0.15/24", mac="00:00:00:00:00:15"),
        "h_attack": net.addHost("h_attack", ip="10.0.0.20/24", mac="00:00:00:00:00:20"),
        "h_attack2": net.addHost(
            "h_attack2", ip="10.0.0.21/24", mac="00:00:00:00:00:21"
        ),
    }

    lk = dict(delay=LINK_DELAY)
    # Core to Agg
    net.addLink(s_core, s_agg1, **lk)
    net.addLink(s_core, s_agg2, **lk)
    # Agg to Access
    net.addLink(s_agg1, s_user_a, **lk)
    net.addLink(s_agg1, s_user_b, **lk)
    net.addLink(s_agg2, s_server, **lk)
    net.addLink(s_agg2, s_dmz, **lk)
    # Hosts to Access
    net.addLink(hosts["h1"], s_user_a, **lk)
    net.addLink(hosts["h2"], s_user_a, **lk)
    net.addLink(hosts["h3"], s_user_a, **lk)
    net.addLink(hosts["h4"], s_user_b, **lk)
    net.addLink(hosts["h5"], s_user_b, **lk)
    net.addLink(hosts["web_srv"], s_server, **lk)
    net.addLink(hosts["db_srv"], s_server, **lk)
    net.addLink(hosts["api_srv"], s_server, **lk)
    net.addLink(hosts["dns_srv"], s_dmz, **lk)
    net.addLink(hosts["h_attack"], s_dmz, **lk)
    net.addLink(hosts["h_attack2"], s_dmz, **lk)

    return net


def run_one(config_key, run_dir):
    cfg = MODEL_CONFIGS[config_key]
    log_path = os.path.join(run_dir, f"ryu_{config_key}.log")
    patch_model_info(config_key)

    ryu = RyuProcess(log_path)
    ryu.start()

    experiment_start = time.time()
    tailer = LogTailer(log_path, experiment_start)
    tailer.start()

    log("  Building Multi-Tier Mininet topology...")
    net = build_topology()
    net.start()

    h1, h2, h3, h4 = net.get("h1"), net.get("h2"), net.get("h3"), net.get("h4")
    web_srv, db_srv, api_srv = net.get("web_srv"), net.get("db_srv"), net.get("api_srv")
    h_attack = net.get("h_attack")

    log("  Warming up (pingall)...")
    net.pingAll(timeout=3)
    time.sleep(2)

    pt = {"normal_start": round(time.time() - experiment_start, 2)}
    run_total = PHASE_NORMAL_S + PHASE_ATTACK_S + PHASE_RECOVERY_S

    log(f"  [t={pt['normal_start']:.0f}s] Phase 1: NORMAL ({PHASE_NORMAL_S}s)")
    web_srv.cmd("iperf -s &")
    db_srv.cmd("iperf -s &")
    api_srv.cmd("iperf -s &")

    # Generate background normal traffic across different nodes
    h1.cmd(f"iperf -c 10.0.0.10 -t {run_total} &")
    h2.cmd(f"iperf -c 10.0.0.11 -t {run_total} &")
    h3.cmd(f"iperf -c 10.0.0.12 -t {run_total} &")
    h4.cmd(f"iperf -c 10.0.0.10 -t {run_total} &")
    time.sleep(PHASE_NORMAL_S)

    pt["attack_start"] = round(time.time() - experiment_start, 2)
    log(f"  [t={pt['attack_start']:.0f}s] Phase 2: ATTACK ({PHASE_ATTACK_S}s)")
    h_attack.cmd("ping -f -s 1400 10.0.0.10 &")

    # Wait for the controller to detect and apply the immediate ban.
    # Because InSDN might take 20-30s to detect, we actively poll for up to 45 seconds.
    log(
        f"  [t={round(time.time() - experiment_start, 2):.0f}s] Waiting up to 45s for network-wide ban to trigger..."
    )

    ban_effective = False
    for i in range(45):
        time.sleep(1)
        check = h_attack.cmd("ping -c 1 -W 1 10.0.0.10")
        if (
            "100% packet loss" in check
            or "100.0% packet loss" in check
            or "0 received" in check
        ):
            ban_effective = True
            log(f"    [INFO] Ban took effect after ~{i + 1} seconds!")
            break

    if not ban_effective:
        log("    [WARN] Ban did not trigger within 45s polling window.")

    # --- ACTIVE BAN VERIFICATION TESTS ---
    log(
        f"  [t={round(time.time() - experiment_start, 2):.0f}s] --- Verifying Immediate Network-Wide Ban ---"
    )

    # Test 1: Attacker to original target
    drop_test_1 = h_attack.cmd("ping -c 3 -W 1 10.0.0.10")
    if "0 received" in drop_test_1 or "100% packet loss" in drop_test_1:
        log("    [PASS] h_attack -> web_srv is BLOCKED.")
    else:
        log("    [FAIL] h_attack -> web_srv is NOT blocked.")
        log(f"    (Debug Output: {drop_test_1.strip()})")

    # Test 2: Attacker to a completely different zone (DMZ DNS)
    drop_test_2 = h_attack.cmd("ping -c 3 -W 1 10.0.0.15")
    if "0 received" in drop_test_2 or "100% packet loss" in drop_test_2:
        log("    [PASS] h_attack -> dns_srv is BLOCKED (Network-wide ban confirmed).")
    else:
        log("    [FAIL] h_attack -> dns_srv is NOT blocked.")

    # Test 3: Normal user to original target
    norm_test = h1.cmd("ping -c 3 -W 1 10.0.0.10")
    if "0 received" not in norm_test and "100% packet loss" not in norm_test:
        log("    [PASS] h1 -> web_srv is CONNECTED (Normal traffic unaffected).")
    else:
        log("    [FAIL] h1 -> web_srv is BLOCKED (Collateral damage detected!).")

    log("  ----------------------------------------------------")

    # Wait out the rest of the attack phase accurately
    elapsed_attack_time = time.time() - experiment_start - pt["attack_start"]
    remaining_attack_s = PHASE_ATTACK_S - elapsed_attack_time
    if remaining_attack_s > 0:
        time.sleep(remaining_attack_s)

    pt["recovery_start"] = round(time.time() - experiment_start, 2)
    log(f"  [t={pt['recovery_start']:.0f}s] Phase 3: RECOVERY ({PHASE_RECOVERY_S}s)")
    h_attack.cmd("kill %ping 2>/dev/null; true")

    # Let normal traffic continue, verifying the ban remains during recovery
    time.sleep(PHASE_RECOVERY_S)

    pt["end"] = round(time.time() - experiment_start, 2)
    log(f"  [t={pt['end']:.0f}s] Complete")

    net.stop()
    tailer.stop()
    ryu.stop()
    mn_cleanup()

    events = tailer.events
    return {
        "config_key": config_key,
        "label": cfg["label"],
        "winner_file": cfg["winner_file"],
        "phase_times": pt,
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", choices=["insdn", "mininet", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_BASE_DIR, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    setLogLevel("warning")
    configs = ["insdn", "mininet"] if args.models == "both" else [args.models]
    results = []

    for key in configs:
        try:
            results.append(run_one(key, run_dir))
        except Exception as exc:
            log(f"ERROR ({key}): {exc}")
        finally:
            mn_cleanup()
            time.sleep(4)

    events_path = os.path.join(run_dir, "events.json")
    with open(events_path, "w") as fp:
        json.dump(
            {
                "run_timestamp": ts,
                "phase_durations": {
                    "normal_s": PHASE_NORMAL_S,
                    "attack_s": PHASE_ATTACK_S,
                    "recovery_s": PHASE_RECOVERY_S,
                },
                "runs": results,
            },
            fp,
            indent=2,
        )

    summary = [f"Experiment Summary -- {ts}", "=" * 50]
    for r in results:
        evts = r["events"]
        dets = [e for e in evts if e["type"] == "detection"]
        clsf = [e for e in evts if e["type"] == "classify"]

        # Filter explicitly by the attacker MAC to prevent false positives calculating fake latency
        atk_clsf = [e for e in clsf if e.get("is_attack_flow")]
        atk_dets = [e for e in dets if e.get("is_attack_flow")]

        summary.extend(
            [
                f"\nModel : {r['label']}",
                f"  File    : {r['winner_file']}",
                f"  Events  : {len(clsf)} classify, {len(dets)} detection(s)",
                f"  Attack flow classifies : {len(atk_clsf)}",
            ]
        )

        if atk_dets:
            d = atk_dets[0]
            lat = d["rel_s"] - r["phase_times"]["attack_start"]
            summary.extend(
                [
                    f"  ATTACKER DETECTED at t={d['rel_s']:.1f}s  prob={d['prob']:.3f}",
                    f"  Detection latency : {lat:.1f}s after attack start",
                ]
            )
        elif dets:
            d = dets[0]
            summary.append(
                f"  WARNING: False positive detection only (at t={d['rel_s']:.1f}s) - Attacker not detected."
            )
        else:
            summary.append("  No detection")

    print("\n".join(summary))
    with open(os.path.join(run_dir, "summary.txt"), "w") as fp:
        fp.write("\n".join(summary))


if __name__ == "__main__":
    main()
