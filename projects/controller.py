#!/usr/bin/env python3
"""
ML-Driven Ryu Controller — v3.4 (Hybrid Detection: ML + Heuristic)
CS 8027: Advanced Networking Architecture

Why v3.3 classified the UDP flood as NORMAL (prob=0.000)
─────────────────────────────────────────────────────────
The InSDN dataset was generated on real/GNS3 hardware where DoS floods
produce 100,000–1,000,000+ pps.  Our Mininet topology is hard-limited to
100 Mbps ≈ 6,500 pps for 1470-byte UDP datagrams.  After StandardScaler
(fitted on InSDN rates), 6,500 pps appears indistinguishable from normal
traffic — the Decision Tree leaf it falls into has 0 attack samples, so
predict_proba returns exactly 0.000.  This is a feature-distribution shift
between the training environment (real hardware) and deployment (Mininet).

v3.4 changes
─────────────
1. DELTA statistics: track per-flow snapshots between polling intervals so
   rate features reflect the *current* 5-second window, not a diluted
   cumulative average that shrinks as flows age.

2. HEURISTIC fallback: three independent rules calibrated for 100 Mbps
   Mininet links that fire when the ML model is blind:
     • asymmetric_flood — pps > 3,000 AND bwd_ratio < 5%  (UDP/SYN floods)
     • extreme_rate     — pps > 50,000 regardless of direction (ICMP flood)
     • volume_flood     — pps > 3,000 AND bps > 4 MB/s (volumetric flood)

3. COMBINED decision: flag as attack if ML prob >= DECISION_THRESHOLD *or*
   any heuristic rule fires.  Both detections are logged separately so you
   can see which path triggered.

4. Snapshot GC: stale per-flow snapshots (flows no longer in the stats
   reply) are purged each cycle to prevent unbounded memory growth.
"""

import glob
import os
import time
from collections import namedtuple

import joblib
import numpy as np

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from ryu.lib import hub
from ryu.lib.packet import ethernet, ether_types, packet
from ryu.ofproto import ofproto_v1_3

# ── ML parameters ─────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.50  # standard 0.5 — no approximation penalty
MIN_PKTS_FOR_CLASSIFY = 20  # skip very short / transient flows

# ── Heuristic thresholds (calibrated for 100 Mbps Mininet links) ──────────────
#   Normal TCP iperf at 10 Mbps  ~  856 pps,  1.25 MB/s,  ~50% symmetric
#   UDP flood  at 100 Mbps       ~ 6429 pps,  9.45 MB/s,  ~0%  backward
#   ICMP ping -f at 100 Mbps     ~ 97,656 pps              ~50% symmetric
#   SYN hping3 --flood           ~ 100,000+ pps             ~0%  backward
HEURISTIC_ENABLED = True
HEURISTIC_FLOOD_PPS = 3_000  # > 3 kpps + asymmetric  -> asymmetric_flood
HEURISTIC_EXTREME_PPS = 50_000  # > 50 kpps any direction -> extreme_rate
HEURISTIC_FLOOD_BPS = 4_000_000  # > 4 MB/s + high pps    -> volume_flood
HEURISTIC_ASYMMETRY = 0.05  # bwd/(fwd+bwd) < 5%     -> one-way flow

# ── Timing / flow management ──────────────────────────────────────────────────
MONITOR_INTERVAL_S = 5  # seconds between flow-stats polls
DROP_HARD_TIMEOUT_S = 60  # drop rule expires after 60 s
FLOW_IDLE_TIMEOUT_S = 10  # forwarding rule idle expiry
DROP_COOLDOWN_S = 120  # min gap between re-blocking the same flow

# ── Artefact paths ─────────────────────────────────────────────────────────────
MODEL_DIR = "/app/projects"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# ── Feature list — MUST match OPENFLOW_FEATURES in training script ─────────────
MODEL_FEATURES = [
    "Flow Duration",  # us   - cumulative, from ofp_flow_stats duration fields
    "Tot Fwd Pkts",  # cnt  - ingress packet_count
    "Tot Bwd Pkts",  # cnt  - egress packet_count
    "Flow Byts/s",  # B/s  - DELTA: bytes in last interval / interval_s
    "Flow Pkts/s",  # p/s  - DELTA: pkts in last interval / interval_s
    "Subflow Fwd Pkts",  # cnt  - equals Tot Fwd Pkts for single-subflow OF entries
    "Subflow Bwd Pkts",  # cnt  - equals Tot Bwd Pkts
    "Subflow Fwd Byts",  # B    - ingress byte_count
    "Subflow Bwd Byts",  # B    - egress byte_count
]

# ── Per-flow snapshot for delta computation ────────────────────────────────────
_FlowSnapshot = namedtuple(
    "_FlowSnapshot",
    ["fwd_pkts", "fwd_bytes", "bwd_pkts", "bwd_bytes", "ts"],
)


def _find_model_path(model_dir: str):
    """
    Locate the optimized model file in model_dir.

    Preference: random_forest -> decision_tree -> mlp_neural_net -> first match.
    """
    for name in [
        "optimized_random_forest.joblib",
        "optimized_decision_tree.joblib",
        "optimized_mlp_neural_net.joblib",
    ]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    candidates = sorted(glob.glob(os.path.join(model_dir, "optimized_*.joblib")))
    return candidates[0] if candidates else None


class MLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self._pending_stats = {}
        self._drop_cooldown = {}
        # (dpid, pair_key) -> _FlowSnapshot — updated every poll cycle
        self._flow_snapshots = {}

        self._load_artefacts()
        self.monitor_thread = hub.spawn(self._monitor)
        self.logger.info(
            "MLController v3.4 started | features=%d | threshold=%.2f | "
            "heuristic=%s | min_pkts=%d | poll=%ds",
            len(MODEL_FEATURES),
            DECISION_THRESHOLD,
            "ON" if HEURISTIC_ENABLED else "OFF",
            MIN_PKTS_FOR_CLASSIFY,
            MONITOR_INTERVAL_S,
        )

    # ── Artefact loading ──────────────────────────────────────────────────────

    def _load_artefacts(self):
        self.ml_model = None
        self.scaler = None

        model_path = _find_model_path(MODEL_DIR)
        if model_path is None:
            self.logger.error(
                "No optimized_*.joblib in %s — copy your model there", MODEL_DIR
            )
        else:
            try:
                self.ml_model = joblib.load(model_path)
                self.logger.info("ML model loaded: %s", os.path.basename(model_path))
            except Exception as exc:
                self.logger.error("ML model load failed: %s", exc)

        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.logger.info("Scaler loaded")
        except Exception as exc:
            self.logger.error("Scaler load failed: %s", exc)

    # ── Switch lifecycle ──────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.logger.info("Switch connected: dpid=%s", dp.id)
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)
            self._pending_stats.pop(dp.id, None)
            stale = [k for k in self._flow_snapshots if k[0] == dp.id]
            for k in stale:
                del self._flow_snapshots[k]
            self.logger.info("Switch disconnected: dpid=%s", dp.id)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow: send all unmatched packets to controller."""
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        self._add_flow(
            dp,
            priority=0,
            match=parser.OFPMatch(),
            actions=[parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)],
        )

    # ── Flow mod helper ───────────────────────────────────────────────────────

    def _add_flow(
        self,
        dp,
        priority,
        match,
        actions,
        buffer_id=None,
        idle_timeout=0,
        hard_timeout=0,
    ):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        kw = dict(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        # `if buffer_id:` treats buffer_id=0 as falsy; must use `is not None`
        if buffer_id is not None:
            kw["buffer_id"] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kw))

    # ── Packet-in handler (L2 learning switch) ────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst, src, dpid = eth.dst, eth.src, dp.id
        self.mac_to_port.setdefault(dpid, {})[src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            buf = msg.buffer_id if msg.buffer_id != ofp.OFP_NO_BUFFER else None
            self._add_flow(
                dp,
                1,
                match,
                actions,
                buffer_id=buf,
                idle_timeout=FLOW_IDLE_TIMEOUT_S,
            )
            if msg.buffer_id != ofp.OFP_NO_BUFFER:
                return

        dp.send_msg(
            parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None,
            )
        )

    # ── Monitoring loop ───────────────────────────────────────────────────────

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                # Reset accumulator before requesting new stats.
                # If a previous reply is still in-flight (very unlikely at 5s
                # intervals) the partial data will be lost; acceptable trade-off.
                self._pending_stats[dp.id] = []
                dp.send_msg(
                    dp.ofproto_parser.OFPFlowStatsRequest(
                        dp,
                        flags=0,
                        table_id=dp.ofproto.OFPTT_ALL,
                        out_port=dp.ofproto.OFPP_ANY,
                        out_group=dp.ofproto.OFPG_ANY,
                        cookie=0,
                        cookie_mask=0,
                        match=dp.ofproto_parser.OFPMatch(),
                    )
                )
            hub.sleep(MONITOR_INTERVAL_S)

    # ── Flow stats reply handler ──────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        # Accumulate priority-1 (learned forwarding) flows only.
        # Priority 0 = table-miss; priority 100 = ML-installed drop rules.
        self._pending_stats.setdefault(dpid, []).extend(
            [f for f in ev.msg.body if f.priority == 1]
        )
        # OFPMPF_REPLY_MORE (0x01) is set on all but the last fragment.
        if not (ev.msg.flags & 0x01):
            self._classify_flows(
                ev.msg.datapath,
                self._pending_stats.get(dpid, []),
            )
            self._pending_stats[dpid] = []

    # ── Delta rate computation ────────────────────────────────────────────────

    def _get_interval_rates(
        self,
        dpid,
        pair_key,
        curr_fwd_pkts,
        curr_fwd_bytes,
        curr_bwd_pkts,
        curr_bwd_bytes,
        flow_dur_s=None,
    ):
        """
        Compute per-interval (delta) counts and elapsed time since last poll.

        Why delta and not cumulative / duration_sec?
        ────────────────────────────────────────────
        OpenFlow duration_sec grows for the entire lifetime of the flow rule.
        For a 30-second attack followed by silence, the cumulative rate at
        t = 50 s is only 60% of the actual peak rate — it *dilutes* the
        signal over time.  Delta statistics give the accurate rate for the
        most recent polling window regardless of flow age.

        Parameters
        ----------
        flow_dur_s : float or None
            Actual flow duration from the OVS stat (duration_sec + duration_nsec).
            Used only on the FIRST observation so the denominator matches the
            real elapsed time rather than the fixed MONITOR_INTERVAL_S constant.
            Example: if the attack started 2.25 s into the 5-second poll cycle,
            using 5 s would underestimate the rate by 55%; using 2.25 s gives
            the correct value.

        Returns (d_fwd_pkts, d_fwd_bytes, d_bwd_pkts, d_bwd_bytes, interval_s)
        """
        now = time.time()
        prev = self._flow_snapshots.get((dpid, pair_key))

        self._flow_snapshots[(dpid, pair_key)] = _FlowSnapshot(
            fwd_pkts=curr_fwd_pkts,
            fwd_bytes=curr_fwd_bytes,
            bwd_pkts=curr_bwd_pkts,
            bwd_bytes=curr_bwd_bytes,
            ts=now,
        )

        if prev is None:
            # First observation: prefer actual flow duration so the rate is
            # accurate even when the flow started mid-interval.
            if flow_dur_s is not None and flow_dur_s > 0:
                interval_s = flow_dur_s
            else:
                interval_s = float(MONITOR_INTERVAL_S)
            return (
                curr_fwd_pkts,
                curr_fwd_bytes,
                curr_bwd_pkts,
                curr_bwd_bytes,
                interval_s,
            )

        interval_s = max(now - prev.ts, 1e-9)
        return (
            max(curr_fwd_pkts - prev.fwd_pkts, 0),
            max(curr_fwd_bytes - prev.fwd_bytes, 0),
            max(curr_bwd_pkts - prev.bwd_pkts, 0),
            max(curr_bwd_bytes - prev.bwd_bytes, 0),
            interval_s,
        )

    # ── Heuristic detector ────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_detect(d_fwd_pkts, d_fwd_bytes, d_bwd_pkts, d_bwd_bytes, interval_s):
        """
        Rule-based flood detection calibrated for 100 Mbps Mininet links.

        The ML model (trained on InSDN real hardware at 100k-1M pps) cannot
        detect Mininet-limited floods (~6,500 pps for 1470-byte UDP).
        These three rules catch what the model misses:

          Rule 1: extreme_rate      pps > 50,000 (any direction)
                  -> catches symmetric ICMP flood (ping -f)

          Rule 2: asymmetric_flood  pps > 3,000 AND bwd_ratio < 5%
                  -> catches UDP flood, SYN flood (one-way traffic)

          Rule 3: volume_flood      pps > 3,000 AND bps > 4 MB/s
                  -> catches volumetric floods with some backward traffic

        All thresholds use DELTA values (last polling interval only) so a
        flow that burst-attacked then went quiet does not stay flagged.

        Returns (detected: bool, reason: str)
        """
        total_pkts = d_fwd_pkts + d_bwd_pkts
        if total_pkts < MIN_PKTS_FOR_CLASSIFY:
            return False, ""

        pps = total_pkts / interval_s
        bps = (d_fwd_bytes + d_bwd_bytes) / interval_s
        bwd_ratio = d_bwd_pkts / total_pkts if total_pkts > 0 else 0.0

        if pps > HEURISTIC_EXTREME_PPS:
            return True, f"extreme_rate pps={pps:,.0f} bwd={bwd_ratio:.3f}"

        if pps > HEURISTIC_FLOOD_PPS and bwd_ratio < HEURISTIC_ASYMMETRY:
            return True, f"asymmetric_flood pps={pps:,.0f} bwd={bwd_ratio:.3f}"

        if pps > HEURISTIC_FLOOD_PPS and bps > HEURISTIC_FLOOD_BPS:
            return True, f"volume_flood pps={pps:,.0f} bps={bps/1e6:.1f}MB/s"

        return False, ""

    # ── Build ML feature vector (9 OpenFlow features) ────────────────────────

    @staticmethod
    def _build_feature_vector(fwd_stat, bwd_stat, delta_byts_s, delta_pkts_s):
        """
        Build the 9-element feature vector matching OPENFLOW_FEATURES in the
        training script.

        Rate features (Flow Byts/s, Flow Pkts/s) use DELTA values so the
        model sees the current rate rather than a lifetime average.  All
        other features remain cumulative to match CICFlowMeter's per-flow
        statistics.

        Returns np.ndarray shape (1, 9) ready for scaler.transform().
        """
        fwd_pkts = fwd_stat.packet_count
        fwd_bytes = fwd_stat.byte_count
        dur_us = (
            fwd_stat.duration_sec * 1_000_000 + fwd_stat.duration_nsec / 1_000
        )  # microseconds

        bwd_pkts = bwd_stat.packet_count if bwd_stat else 0
        bwd_bytes = bwd_stat.byte_count if bwd_stat else 0

        return np.array(
            [
                [
                    dur_us,  # Flow Duration      (us, cumulative)
                    fwd_pkts,  # Tot Fwd Pkts        (cumulative)
                    bwd_pkts,  # Tot Bwd Pkts        (cumulative)
                    delta_byts_s,  # Flow Byts/s         (DELTA - current rate)
                    delta_pkts_s,  # Flow Pkts/s         (DELTA - current rate)
                    fwd_pkts,  # Subflow Fwd Pkts    (= Tot Fwd Pkts)
                    bwd_pkts,  # Subflow Bwd Pkts    (= Tot Bwd Pkts)
                    fwd_bytes,  # Subflow Fwd Byts    (cumulative)
                    bwd_bytes,  # Subflow Bwd Byts    (cumulative)
                ]
            ],
            dtype=np.float64,
        )

    # ── ML + heuristic classification ────────────────────────────────────────

    def _classify_flows(self, dp, flow_stats):
        """
        Classify each bidirectional flow seen on datapath dp.

        Detection logic:
          1. Compute delta statistics since the last poll.
          2. Run ML model on the 9-feature vector  -> probability score.
          3. Run heuristic rules on delta statistics.
          4. Flag as attack if ML prob >= DECISION_THRESHOLD OR heuristic fires.
          5. Install drop rules (priority 100, hard_timeout 60s) for both
             directions. Respect DROP_COOLDOWN_S to avoid duplicate installs.
        """
        # ── Build MAC-pair lookup ─────────────────────────────────────────────
        stat_map = {}
        for stat in flow_stats:
            src = stat.match.get("eth_src")
            dst = stat.match.get("eth_dst")
            if src and dst:
                stat_map[(src, dst)] = stat

        active_pairs = set()  # for snapshot GC at end of cycle
        processed = set()

        for (src, dst), fwd_stat in stat_map.items():
            pair_key = frozenset({src, dst})
            if pair_key in processed:
                continue
            processed.add(pair_key)
            active_pairs.add(pair_key)

            bwd_stat = stat_map.get((dst, src))

            # Ensure fwd is the higher-volume direction for consistent features
            if bwd_stat and bwd_stat.packet_count > fwd_stat.packet_count:
                fwd_stat, bwd_stat, src, dst = bwd_stat, fwd_stat, dst, src

            total_pkts = fwd_stat.packet_count + (
                bwd_stat.packet_count if bwd_stat else 0
            )
            if total_pkts < MIN_PKTS_FOR_CLASSIFY:
                continue

            curr_fwd_pkts = fwd_stat.packet_count
            curr_fwd_bytes = fwd_stat.byte_count
            curr_bwd_pkts = bwd_stat.packet_count if bwd_stat else 0
            curr_bwd_bytes = bwd_stat.byte_count if bwd_stat else 0

            # Actual flow duration from OVS stat — used as first-poll denominator
            # so the rate is not underestimated when the flow started mid-interval.
            flow_dur_s = max(
                fwd_stat.duration_sec + fwd_stat.duration_nsec / 1_000_000_000,
                1e-9,
            )

            # ── Delta statistics ──────────────────────────────────────────────
            d_fwd_pkts, d_fwd_bytes, d_bwd_pkts, d_bwd_bytes, interval_s = (
                self._get_interval_rates(
                    dp.id,
                    pair_key,
                    curr_fwd_pkts,
                    curr_fwd_bytes,
                    curr_bwd_pkts,
                    curr_bwd_bytes,
                    flow_dur_s=flow_dur_s,
                )
            )

            d_total_pkts = d_fwd_pkts + d_bwd_pkts
            d_total_bytes = d_fwd_bytes + d_bwd_bytes
            delta_pkts_s = d_total_pkts / interval_s
            delta_byts_s = d_total_bytes / interval_s

            # ── ML inference ──────────────────────────────────────────────────
            ml_prob = 0.0
            ml_available = bool(self.ml_model and self.scaler)
            if ml_available:
                try:
                    features = self._build_feature_vector(
                        fwd_stat, bwd_stat, delta_byts_s, delta_pkts_s
                    )
                    features_s = self.scaler.transform(features)
                    ml_prob = self.ml_model.predict_proba(features_s)[0][1]
                except Exception as exc:
                    self.logger.error("Inference error %s->%s: %s", src, dst, exc)

            # ── Heuristic inference ───────────────────────────────────────────
            heuristic_hit, heuristic_reason = False, ""
            if HEURISTIC_ENABLED:
                heuristic_hit, heuristic_reason = self._heuristic_detect(
                    d_fwd_pkts,
                    d_fwd_bytes,
                    d_bwd_pkts,
                    d_bwd_bytes,
                    interval_s,
                )

            # ── Combined decision ─────────────────────────────────────────────
            ml_attack = ml_prob >= DECISION_THRESHOLD
            is_attack = ml_attack or heuristic_hit

            flow_key = f"{src}->{dst}"
            now = time.time()

            if is_attack:
                last_block = self._drop_cooldown.get((dp.id, flow_key), 0)
                if now - last_block < DROP_COOLDOWN_S:
                    continue
                self._drop_cooldown[(dp.id, flow_key)] = now

                if ml_attack and heuristic_hit:
                    det = f"ML({ml_prob:.3f})+HEURISTIC({heuristic_reason})"
                elif ml_attack:
                    det = f"ML({ml_prob:.3f})"
                else:
                    det = f"HEURISTIC({heuristic_reason})"

                self.logger.warning(
                    "ATTACK BLOCK | dpid=%-4s | %-19s -> %-19s | "
                    "det=%s | dpps=%.0f | pkts=%d",
                    dp.id,
                    src,
                    dst,
                    det,
                    delta_pkts_s,
                    total_pkts,
                )

                self._add_flow(
                    dp,
                    100,
                    fwd_stat.match,
                    [],
                    hard_timeout=DROP_HARD_TIMEOUT_S,
                )
                if bwd_stat:
                    self._add_flow(
                        dp,
                        100,
                        bwd_stat.match,
                        [],
                        hard_timeout=DROP_HARD_TIMEOUT_S,
                    )
            else:
                self.logger.info(
                    "NORMAL       | dpid=%-4s | %-19s -> %-19s | "
                    "ML=%.3f | dpps=%.0f | pkts=%d",
                    dp.id,
                    src,
                    dst,
                    ml_prob,
                    delta_pkts_s,
                    total_pkts,
                )

        # ── Snapshot garbage collection ───────────────────────────────────────
        stale = [
            k
            for k in self._flow_snapshots
            if k[0] == dp.id and k[1] not in active_pairs
        ]
        for k in stale:
            del self._flow_snapshots[k]