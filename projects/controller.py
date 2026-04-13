import glob
import os
from collections import defaultdict

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
# CONFIG
# ─────────────────────────────────────────────
DECISION_THRESHOLD = 0.45
MIN_PKTS_FOR_CLASSIFY = 5
MONITOR_INTERVAL_S = 2
DROP_HARD_TIMEOUT_S = 60
FLOW_IDLE_TIMEOUT_S = 10

MODEL_DIR = "/app/projects"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


def _find_model():
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "optimized_*.joblib")))
    return files[0] if files else None


# ─────────────────────────────────────────────
# CONTROLLER
# ─────────────────────────────────────────────
class MLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datapaths = {}
        self.mac_to_port = {}
        self.detected_flows = set()

        self._load_model()
        hub.spawn(self._monitor)

        self.logger.info("🚀 ML Controller Ready")

    # ─────────────────────────────────────────────
    # LOAD MODEL
    # ─────────────────────────────────────────────
    def _load_model(self):
        try:
            model_path = _find_model()
            self.model = joblib.load(model_path)
            self.logger.info(f"✅ Model: {model_path}")
        except Exception as e:
            self.logger.error(f"❌ Model load failed: {e}")
            self.model = None

        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.logger.info("✅ Scaler loaded")
        except Exception as e:
            self.logger.error(f"❌ Scaler load failed: {e}")
            self.scaler = None

    # ─────────────────────────────────────────────
    # SWITCH EVENTS
    # ─────────────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

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

    # ─────────────────────────────────────────────
    # PACKET SWITCHING
    # ─────────────────────────────────────────────
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

    # ─────────────────────────────────────────────
    # MONITOR
    # ─────────────────────────────────────────────
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            hub.sleep(MONITOR_INTERVAL_S)

    # ─────────────────────────────────────────────
    # FLOW STATS
    # ─────────────────────────────────────────────
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def stats_reply(self, ev):
        dp = ev.msg.datapath
        flows = [f for f in ev.msg.body if f.priority >= 1]
        self._classify(dp, flows)

    # ─────────────────────────────────────────────
    # FEATURE BUILDER
    # ─────────────────────────────────────────────
    def _features(self, fwd, bwd):
        fwd_pkts = fwd.packet_count
        fwd_bytes = fwd.byte_count

        bwd_pkts = bwd.packet_count if bwd else 0
        bwd_bytes = bwd.byte_count if bwd else 0

        duration = max(fwd.duration_sec + fwd.duration_nsec / 1e9, 1e-9)

        total_bytes = fwd_bytes + bwd_bytes
        total_pkts = fwd_pkts + bwd_pkts

        return np.array(
            [
                [
                    duration,
                    fwd_pkts,
                    bwd_pkts,
                    total_bytes / duration,
                    total_pkts / duration,
                    fwd_pkts,
                    bwd_pkts,
                    fwd_bytes,
                    bwd_bytes,
                ]
            ]
        )

    # ─────────────────────────────────────────────
    # DELETE FLOW (WINDOW RESET)
    # ─────────────────────────────────────────────
    def _delete_flow(self, dp, src, dst):
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

    # ─────────────────────────────────────────────
    # CLASSIFICATION
    # ─────────────────────────────────────────────
    def _classify(self, dp, flows):

        stat_map = {}
        for f in flows:
            src = f.match.get("eth_src")
            dst = f.match.get("eth_dst")
            if src and dst:
                stat_map[(src, dst)] = f

        processed = set()

        for (src, dst), fwd in stat_map.items():
            key = frozenset([src, dst])
            if key in processed:
                continue
            processed.add(key)

            bwd = stat_map.get((dst, src))
            total_pkts = fwd.packet_count + (bwd.packet_count if bwd else 0)

            if total_pkts < MIN_PKTS_FOR_CLASSIFY:
                continue

            try:
                X = self.scaler.transform(self._features(fwd, bwd))

                if hasattr(self.model, "predict_proba"):
                    prob = self.model.predict_proba(X)[0][1]
                else:
                    prob = float(self.model.predict(X)[0])

                self.logger.info(f"{src}->{dst} pkts={total_pkts} prob={prob:.3f}")

            except Exception as e:
                self.logger.error(f"Inference error: {e}")
                continue

            flow_id = (src, dst)

            if prob >= DECISION_THRESHOLD and flow_id not in self.detected_flows:
                self.detected_flows.add(flow_id)

                self.logger.warning(f"🚨 ATTACK {src}->{dst}")

                match = dp.ofproto_parser.OFPMatch(eth_src=src, eth_dst=dst)

                # DROP RULE
                self._add_flow(dp, 100, match, [], hard=DROP_HARD_TIMEOUT_S)

                # 🔥 CRITICAL: RESET FLOW FOR FAST DETECTION
                self._delete_flow(dp, src, dst)