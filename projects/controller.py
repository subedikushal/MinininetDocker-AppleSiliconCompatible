"""
controller.py — ML-Driven Ryu SDN Intrusion Detection Controller
CS 8027: Advanced Networking Architecture

Design principles
─────────────────
• NO StandardScaler — all 4 candidate models are tree-based and
  scale-invariant.  Raw feature values are fed directly to the model,
  exactly as they were during training.

• Flow Duration in SECONDS — the training notebook converts CICFlowMeter's
  µs values to seconds before fitting, so OpenFlow's
  `duration_sec + duration_nsec/1e9` can be used directly at inference.
  No multiplication by 1e6 is needed anywhere in this file.

• Model selection via model_info.json — the training notebook saves a
  JSON manifest that records which .joblib file is the winner, what
  features to use, and what decision threshold to apply.  The controller
  reads this file at startup so no hardcoded model name is needed.

• Automatic re-detection — detected_flows stores a timestamp; entries
  expire after DROP_HARD_TIMEOUT_S so a resumed attack is caught again.

File layout expected at MODEL_DIR (/app/projects/):
    model_info.json          ← manifest written by the notebook
    model_rf.joblib          ← Random Forest
    model_dt.joblib          ← Decision Tree
    model_xgb.joblib         ← XGBoost (sklearn wrapper)
    model_xgb.json           ← XGBoost (native, not loaded by this controller)
    model_lgbm.joblib        ← LightGBM
"""

import json
import os
import time
from typing import Dict, Optional, Tuple

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

# ─────────────────────────────────────────────
# CONFIG  (can be overridden via environment variables)
# ─────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/projects")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

# Fallback values — the model_info.json overrides these at runtime
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.45"))
MIN_PKTS_FOR_CLASSIFY = int(os.environ.get("MIN_PKTS", "5"))

MONITOR_INTERVAL_S = 2  # seconds between flow-stats polls
DROP_HARD_TIMEOUT_S = 60  # seconds a drop rule lives before expiring
FLOW_IDLE_TIMEOUT_S = 10  # seconds before an idle forwarding flow is removed


# ─────────────────────────────────────────────
# CONTROLLER
# ─────────────────────────────────────────────
class MLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datapaths = {}  # dpid → datapath
        self.mac_to_port = {}  # dpid → {mac → port}

        # Flow re-detection: {(src_mac, dst_mac) → timestamp_of_detection}
        # Entries are evicted by _monitor() after DROP_HARD_TIMEOUT_S so the
        # same pair can be re-detected if the attack resumes.
        self.detected_flows = {}  # type: Dict[Tuple, float]

        self._load_model()
        hub.spawn(self._monitor)

        self.logger.info("ML Controller ready")

    # ──────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────
    def _load_model(self):
        """
        Load model_info.json first to find the winner model file, then
        load that .joblib.  No scaler is loaded — none exists.
        """
        global DECISION_THRESHOLD

        # Step 1: read the manifest
        try:
            with open(MODEL_INFO_FILE) as fp:
                info = json.load(fp)
            winner_file = os.path.join(MODEL_DIR, info["winner_file"])
            DECISION_THRESHOLD = float(
                info.get("decision_threshold", DECISION_THRESHOLD)
            )
            self.feature_names = info.get("feature_names", [])
            self.logger.info(
                "Manifest loaded: winner=%s  threshold=%.2f",
                info["winner"],
                DECISION_THRESHOLD,
            )
        except Exception as exc:
            self.logger.error(
                "Cannot read %s: %s  "
                "→ Ensure the training notebook has been run and "
                "model_info.json is present in %s",
                MODEL_INFO_FILE,
                exc,
                MODEL_DIR,
            )
            self.model = None
            return

        # Step 2: load the winner model
        try:
            self.model = joblib.load(winner_file)
            self.logger.info(
                "Model loaded: %s  (%s)",
                winner_file,
                type(self.model).__name__,
            )
        except Exception as exc:
            self.logger.error("Model load failed (%s): %s", winner_file, exc)
            self.model = None
            return

        # Step 3: verify the model has predict_proba
        if not hasattr(self.model, "predict_proba"):
            self.logger.warning(
                "Model %s has no predict_proba; will use predict() as binary.",
                type(self.model).__name__,
            )

        self.logger.info(
            "Inference pipeline READY — NO scaler, raw features in seconds/counts/bytes"
        )

    # ──────────────────────────────────────────
    # SWITCH EVENTS
    # ──────────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.logger.info("Switch connected    dpid=%016x", dp.id)
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)
            # dp.id can be None if the switch disconnected before handshake
            dpid_str = ("%016x" % dp.id) if dp.id is not None else "unknown"
            self.logger.info("Switch disconnected dpid=%s", dpid_str)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        # Table-miss: send everything to the controller
        self._add_flow(
            dp,
            0,
            parser.OFPMatch(),
            [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)],
        )

    def _add_flow(self, dp, priority, match, actions, idle=0, hard=0):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle,
            hard_timeout=hard,
        )
        dp.send_msg(mod)

    # ──────────────────────────────────────────
    # PACKET SWITCHING
    # ──────────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        src = eth.src
        dst = eth.dst
        dpid = dp.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = msg.match["in_port"]

        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=msg.match["in_port"], eth_src=src, eth_dst=dst
            )
            self._add_flow(dp, 1, match, actions, idle=FLOW_IDLE_TIMEOUT_S)

        dp.send_msg(
            parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=msg.match["in_port"],
                actions=actions,
                data=msg.data,
            )
        )

    # ──────────────────────────────────────────
    # MONITOR  (polls stats + cleans up detected_flows)
    # ──────────────────────────────────────────
    def _monitor(self):
        while True:
            # Snapshot to avoid dict-size changes during iteration
            for dp in list(self.datapaths.values()):
                req = dp.ofproto_parser.OFPFlowStatsRequest(dp, 0, dp.ofproto.OFPTT_ALL)
                dp.send_msg(req)

            # Evict stale detections so the same flow can be re-detected
            # after its DROP rule has expired (DROP_HARD_TIMEOUT_S seconds)
            now = time.time()
            for fid in [
                k
                for k, ts in list(self.detected_flows.items())
                if now - ts > DROP_HARD_TIMEOUT_S
            ]:
                del self.detected_flows[fid]
                self.logger.info(
                    "Cleared detection for %s after %ds — re-detection enabled",
                    fid,
                    DROP_HARD_TIMEOUT_S,
                )

            hub.sleep(MONITOR_INTERVAL_S)

    # ──────────────────────────────────────────
    # STATS REPLY
    # ──────────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def stats_reply(self, ev):
        dp = ev.msg.datapath
        flows = [f for f in ev.msg.body if f.priority >= 1]
        self._classify(dp, flows)

    # ──────────────────────────────────────────
    # FEATURE EXTRACTION
    # ──────────────────────────────────────────
    def _build_feature_vector(self, fwd, bwd):
        """
        Build the 9-element raw feature vector matching the training data.

        Feature index and training-time unit (all identical to what we compute here):
          [0] Flow Duration    — seconds  (training: µs÷1e6; OpenFlow: duration_sec)
          [1] Tot Fwd Pkts     — count
          [2] Tot Bwd Pkts     — count
          [3] Flow Byts/s      — bytes/second
          [4] Flow Pkts/s      — packets/second
          [5] Subflow Fwd Pkts — count  (≡ Tot Fwd Pkts, r=1.000 in training data)
          [6] Subflow Bwd Pkts — count  (≡ Tot Bwd Pkts, r=1.000 in training data)
          [7] Subflow Fwd Byts — bytes
          [8] Subflow Bwd Byts — bytes

        NO scaling is applied — the tree models are trained on raw values
        and their split thresholds are already in the units shown above.
        """
        fwd_pkts = fwd.packet_count
        fwd_bytes = fwd.byte_count

        bwd_pkts = bwd.packet_count if bwd else 0
        bwd_bytes = bwd.byte_count if bwd else 0

        # duration in SECONDS — matches training (notebook divided µs by 1e6)
        duration_s = max(fwd.duration_sec + fwd.duration_nsec / 1e9, 1e-9)

        total_bytes = fwd_bytes + bwd_bytes
        total_pkts = fwd_pkts + bwd_pkts

        return np.array(
            [
                [
                    duration_s,  # [0] Flow Duration (s)
                    fwd_pkts,  # [1] Tot Fwd Pkts
                    bwd_pkts,  # [2] Tot Bwd Pkts
                    total_bytes / duration_s,  # [3] Flow Byts/s
                    total_pkts / duration_s,  # [4] Flow Pkts/s
                    fwd_pkts,  # [5] Subflow Fwd Pkts
                    bwd_pkts,  # [6] Subflow Bwd Pkts
                    fwd_bytes,  # [7] Subflow Fwd Byts
                    bwd_bytes,  # [8] Subflow Bwd Byts
                ]
            ]
        )

    # ──────────────────────────────────────────
    # DROP HELPERS
    # ──────────────────────────────────────────
    def _drop_flow(self, dp, src, dst):
        """Install a high-priority drop rule for src→dst traffic."""
        match = dp.ofproto_parser.OFPMatch(eth_src=src, eth_dst=dst)
        # Priority 100 beats the forwarding rule (priority 1)
        self._add_flow(dp, 100, match, [], hard=DROP_HARD_TIMEOUT_S)

    def _delete_forwarding_entry(self, dp, src, dst):
        """
        Remove the existing forwarding entry so the drop rule takes effect
        immediately instead of waiting for the forwarding rule's idle timeout.
        """
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        match = parser.OFPMatch(eth_src=src, eth_dst=dst)
        mod = parser.OFPFlowMod(
            datapath=dp,
            command=ofp.OFPFC_DELETE,
            out_port=ofp.OFPP_ANY,
            out_group=ofp.OFPG_ANY,
            match=match,
        )
        dp.send_msg(mod)

    # ──────────────────────────────────────────
    # CLASSIFICATION
    # ──────────────────────────────────────────
    def _classify(self, dp, flows):
        if self.model is None:
            self.logger.warning(
                "Inference skipped: model not loaded. "
                "Check that model_info.json and the winner .joblib exist in %s",
                MODEL_DIR,
            )
            return

        # Build (src, dst) → flow-stats lookup
        stat_map = {}
        for f in flows:
            src = f.match.get("eth_src")
            dst = f.match.get("eth_dst")
            if src and dst:
                stat_map[(src, dst)] = f

        processed = set()

        for (src, dst), fwd in list(stat_map.items()):
            # Process each bidirectional pair once
            key = frozenset([src, dst])
            if key in processed:
                continue
            processed.add(key)

            bwd = stat_map.get((dst, src))
            total_pkts = fwd.packet_count + (bwd.packet_count if bwd else 0)

            if total_pkts < MIN_PKTS_FOR_CLASSIFY:
                continue

            try:
                # Raw feature vector — NO scaler transform
                X = self._build_feature_vector(fwd, bwd)

                if hasattr(self.model, "predict_proba"):
                    prob = self.model.predict_proba(X)[0][1]
                else:
                    prob = float(self.model.predict(X)[0])

            except Exception as exc:
                self.logger.error("Inference error %s→%s: %s", src, dst, exc)
                continue

            self.logger.info(
                "[CLASSIFY] %s→%s  pkts=%d  dur_s=%.2f  "
                "byts_s=%.0f  bwd_bytes=%d  prob=%.3f",
                src,
                dst,
                total_pkts,
                X[0][0],  # Flow Duration (s)
                X[0][3],  # Flow Byts/s
                int(X[0][8]),  # Subflow Bwd Byts — key diagnostic field
                prob,
            )

            flow_id = (src, dst)

            if prob >= DECISION_THRESHOLD and flow_id not in self.detected_flows:
                self.detected_flows[flow_id] = time.time()

                self.logger.warning(
                    "ATTACK DETECTED  %s→%s  prob=%.3f  dpid=%016x",
                    src,
                    dst,
                    prob,
                    dp.id,
                )

                # 1. Install drop rule (priority 100, expires after 60 s)
                self._drop_flow(dp, src, dst)

                # 2. Delete existing forwarding rule so drop takes effect NOW
                self._delete_forwarding_entry(dp, src, dst)