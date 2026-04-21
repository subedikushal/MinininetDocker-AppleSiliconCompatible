import json
import os
import time

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
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/projects/models")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

# Fallback values — the model_info.json overrides these at runtime
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
MIN_PKTS_FOR_CLASSIFY = int(os.environ.get("MIN_PKTS", "5"))

MONITOR_INTERVAL_S = 2  # seconds between flow-stats polls
DROP_HARD_TIMEOUT_S = 60  # seconds a specific flow drop rule lives before expiring
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
        self.detected_flows = {}

        # --- NEW: Track MACs that are scheduled for a complete ban ---
        self.banned_macs = set()

        self._load_model()
        hub.spawn(self._monitor)

        self.logger.info("ML Controller ready")

    # ──────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────
    def _load_model(self):
        global DECISION_THRESHOLD

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
                "Cannot read %s: %s  → Ensure model_info.json is present in %s",
                MODEL_INFO_FILE,
                exc,
                MODEL_DIR,
            )
            self.model = None
            return

        try:
            self.model = joblib.load(winner_file)
            self.logger.info(
                "Model loaded: %s (%s)", winner_file, type(self.model).__name__
            )
        except Exception as exc:
            self.logger.error("Model load failed (%s): %s", winner_file, exc)
            self.model = None
            return

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
            dpid_str = ("%016x" % dp.id) if dp.id is not None else "unknown"
            self.logger.info("Switch disconnected dpid=%s", dpid_str)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
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
    # MONITORING
    # ──────────────────────────────────────────
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                req = dp.ofproto_parser.OFPFlowStatsRequest(dp, 0, dp.ofproto.OFPTT_ALL)
                dp.send_msg(req)

            now = time.time()
            for fid in [
                k
                for k, ts in list(self.detected_flows.items())
                if now - ts > DROP_HARD_TIMEOUT_S
            ]:
                del self.detected_flows[fid]

            hub.sleep(MONITOR_INTERVAL_S)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def stats_reply(self, ev):
        dp = ev.msg.datapath
        flows = [
            f for f in ev.msg.body if f.priority == 1
        ]  # Only check forwarding rules
        self._classify(dp, flows)

    def _build_feature_vector(self, fwd, bwd):
        fwd_pkts = fwd.packet_count
        fwd_bytes = fwd.byte_count
        bwd_pkts = bwd.packet_count if bwd else 0
        bwd_bytes = bwd.byte_count if bwd else 0
        duration_s = max(fwd.duration_sec + fwd.duration_nsec / 1e9, 1e-9)

        total_bytes = fwd_bytes + bwd_bytes
        total_pkts = fwd_pkts + bwd_pkts

        return np.array(
            [
                [
                    duration_s,
                    fwd_pkts,
                    bwd_pkts,
                    total_bytes / duration_s,
                    total_pkts / duration_s,
                    fwd_pkts,
                    bwd_pkts,
                    fwd_bytes,
                    bwd_bytes,
                ]
            ]
        )

    # ──────────────────────────────────────────
    # MITIGATION HELPERS
    # ──────────────────────────────────────────
    def _drop_flow(self, dp, src, dst):
        """Immediate flow-specific drop (Priority 100)"""
        try:
            match = dp.ofproto_parser.OFPMatch(eth_src=src, eth_dst=dst)
            self._add_flow(dp, 100, match, [], hard=DROP_HARD_TIMEOUT_S)
        except Exception as e:
            self.logger.error("Failed to drop flow %s->%s: %s", src, dst, e)

    def _delete_forwarding_entry(self, dp, src=None, dst=None):
        try:
            parser = dp.ofproto_parser
            ofp = dp.ofproto

            kwargs = {}
            if src:
                kwargs["eth_src"] = src
            if dst:
                kwargs["eth_dst"] = dst

            match = parser.OFPMatch(**kwargs)
            mod = parser.OFPFlowMod(
                datapath=dp,
                command=ofp.OFPFC_DELETE,
                table_id=ofp.OFPTT_ALL,
                out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY,
                match=match,
            )
            dp.send_msg(mod)
        except Exception as e:
            self.logger.error("Error deleting entry %s -> %s: %s", src, dst, e)

    # ──────────────────────────────────────────
    # NETWORK-WIDE BAN
    # ──────────────────────────────────────────
    def _ban_mac_network_wide(self, src_mac):
        """Install permanent drop rules on ALL switches immediately"""
        self.logger.error("Executing immediate network-wide BLOCK on MAC: %s", src_mac)

        for dp in list(self.datapaths.values()):
            try:
                parser = dp.ofproto_parser

                # 1. Drop all traffic FROM the attacker
                match_src = parser.OFPMatch(eth_src=src_mac)
                self._add_flow(dp, 200, match_src, [], idle=0, hard=0)

                # 2. Drop all traffic TO the attacker
                match_dst = parser.OFPMatch(eth_dst=src_mac)
                self._add_flow(dp, 200, match_dst, [], idle=0, hard=0)

                # 3. Clean up existing forwarding entries to prevent fast-path caching
                self._delete_forwarding_entry(dp, src=src_mac)
                self._delete_forwarding_entry(dp, dst=src_mac)

            except Exception as e:
                self.logger.error(
                    "Failed to execute network ban on dp %s: %s", dp.id, e
                )

        # Purge from mac_to_port cache
        for dpid in self.mac_to_port:
            if src_mac in self.mac_to_port[dpid]:
                del self.mac_to_port[dpid][src_mac]

    # ──────────────────────────────────────────
    # CLASSIFICATION
    # ──────────────────────────────────────────
    def _classify(self, dp, flows):
        if self.model is None:
            return

        stat_map = {}
        for f in flows:
            src = f.match.get("eth_src")
            dst = f.match.get("eth_dst")
            if src and dst:
                stat_map[(src, dst)] = f

        processed = set()
        for (src, dst), fwd in list(stat_map.items()):
            key = frozenset([src, dst])
            if key in processed:
                continue
            processed.add(key)

            bwd = stat_map.get((dst, src))
            total_pkts = fwd.packet_count + (bwd.packet_count if bwd else 0)

            if total_pkts < MIN_PKTS_FOR_CLASSIFY:
                continue

            try:
                X = self._build_feature_vector(fwd, bwd)
                prob = (
                    self.model.predict_proba(X)[0][1]
                    if hasattr(self.model, "predict_proba")
                    else float(self.model.predict(X)[0])
                )
            except Exception as exc:
                continue

            self.logger.info(
                "[CLASSIFY] %s→%s  pkts=%d  dur_s=%.2f  byts_s=%.0f  bwd_bytes=%d  prob=%.3f",
                src,
                dst,
                total_pkts,
                X[0][0],
                X[0][3],
                int(X[0][8]),
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

                # 1. Immediately drop this specific attack flow
                self._drop_flow(dp, src, dst)
                self._delete_forwarding_entry(dp, src=src, dst=dst)

                # 2. Execute the complete network-wide block of the attacker immediately
                if src not in self.banned_macs:
                    self.banned_macs.add(src)
                    self._ban_mac_network_wide(src)
