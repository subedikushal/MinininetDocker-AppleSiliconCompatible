#!/usr/bin/env python3
"""
topology.py — 3-Zone SDN Topology for ML Intrusion Detection
CS 8027: Advanced Networking Architecture

Topology overview
─────────────────
                    ┌─────────┐
                    │  s4     │  ← Core / backbone switch
                    │ (core)  │
                    └────┬────┘
            ┌────────────┼────────────┐
            │            │            │
       ┌────┴────┐  ┌────┴────┐  ┌───┴─────┐
       │   s1    │  │   s2    │  │   s3    │
       │ (user)  │  │(server) │  │(attack) │
       └─┬──┬──┬─┘  └───┬──┬─┘  └────┬────┘
         │  │  │        │  │          │
        h1 h2 h3    web_srv db_srv  h_attack

All links are UNLIMITED bandwidth (no TCLink shaping).
A small propagation delay (2 ms) simulates realistic link latency.

Host map
────────
  h1          10.0.0.1/24   User zone    s1
  h2          10.0.0.2/24   User zone    s1
  h3          10.0.0.3/24   User zone    s1
  web_srv     10.0.0.10/24  Server zone  s2
  db_srv      10.0.0.11/24  Server zone  s2
  h_attack    10.0.0.20/24  Attack zone  s3

Controller: RemoteController at 127.0.0.1:6653  (Ryu)
"""

import argparse

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController

# ── Tunables ──────────────────────────────────────────────────────────────────
CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 6653
LINK_DELAY = "2ms"  # propagation delay; no bandwidth cap


def build_and_run(controller_ip=CONTROLLER_IP):
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # ── Controller ────────────────────────────────────────────────────────────
    net.addController(
        "c0",
        controller=RemoteController,
        ip=controller_ip,
        port=CONTROLLER_PORT,
    )

    # ── Switches ──────────────────────────────────────────────────────────────
    s_user = net.addSwitch("s1", protocols="OpenFlow13")  # user zone
    s_server = net.addSwitch("s2", protocols="OpenFlow13")  # server zone
    s_attack = net.addSwitch("s3", protocols="OpenFlow13")  # attacker zone
    s_core = net.addSwitch("s4", protocols="OpenFlow13")  # backbone

    # ── Hosts ─────────────────────────────────────────────────────────────────
    h1 = net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")
    h3 = net.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03")
    web_srv = net.addHost("web_srv", ip="10.0.0.10/24", mac="00:00:00:00:00:10")
    db_srv = net.addHost("db_srv", ip="10.0.0.11/24", mac="00:00:00:00:00:11")
    h_attack = net.addHost("h_attack", ip="10.0.0.20/24", mac="00:00:00:00:00:20")

    # ── Links (delay only, no bandwidth limit) ────────────────────────────────
    lk = dict(delay=LINK_DELAY)

    # Hosts → access switches
    net.addLink(h1, s_user, **lk)
    net.addLink(h2, s_user, **lk)
    net.addLink(h3, s_user, **lk)
    net.addLink(web_srv, s_server, **lk)
    net.addLink(db_srv, s_server, **lk)
    net.addLink(h_attack, s_attack, **lk)

    # Access switches → core
    net.addLink(s_user, s_core, **lk)
    net.addLink(s_server, s_core, **lk)
    net.addLink(s_attack, s_core, **lk)

    # ── Start ─────────────────────────────────────────────────────────────────
    info("*** Starting network\n")
    net.start()
    _print_guide()
    CLI(net)
    net.stop()


def _print_guide():
    sep = "═" * 68
    info(f"\n{sep}\n")
    info("  3-Zone SDN ML Intrusion Detection — Experiment Guide\n")
    info(f"{sep}\n")
    info(
        """
  ┌──────────────────────────────────────────────────────────────────┐
  │ Host       IP           Zone      Switch  Notes                  │
  ├──────────────────────────────────────────────────────────────────┤
  │ h1         10.0.0.1     User      s1      Normal client          │
  │ h2         10.0.0.2     User      s1      Normal client          │
  │ h3         10.0.0.3     User      s1      Normal client          │
  │ web_srv    10.0.0.10    Server    s2      HTTP / iperf target     │
  │ db_srv     10.0.0.11    Server    s2      DB / iperf target       │
  │ h_attack   10.0.0.20    Attacker  s3      Attack source          │
  └──────────────────────────────────────────────────────────────────┘

  STEP 1 — Verify connectivity (expect 0% drop)
  ──────────────────────────────────────────────
  mininet> pingall

  STEP 2 — Normal traffic  [controller must NOT alert]
  ─────────────────────────────────────────────────────
  mininet> web_srv iperf -s &
  mininet> db_srv  iperf -s &
  mininet> h1 iperf -c 10.0.0.10 -t 30 &
  mininet> h2 iperf -c 10.0.0.11 -t 30 &
  mininet> h3 iperf -c 10.0.0.10 -t 20 &

  STEP 3 — Attack simulation  [controller MUST alert + drop]
  ────────────────────────────────────────────────────────────
  [A] ICMP flood — large packets, symmetric (best for detection)
      mininet> h_attack ping -f -s 1400 10.0.0.10

  [B] SYN flood — half-open connections (requires hping3)
      mininet> h_attack hping3 -S --flood -p 80 10.0.0.10

  [C] UDP flood WITH server running (so bwd bytes are generated)
      mininet> web_srv iperf -s -u &
      mininet> h_attack iperf -c 10.0.0.10 -u -b 1000M -t 30

  [D] Bidirectional TCP flood
      mininet> web_srv iperf -s &
      mininet> h_attack iperf -c 10.0.0.10 -t 30 -d

  NOTE: The model's most important feature is Subflow_Bwd_Byts (bytes
  flowing back from the server). Attacks that generate NO server
  response (e.g. UDP flood with no listener) will score ~0 probability
  because the model was trained on attacks that produce server-side
  traffic. Use option [A], [B], or [C] for reliable detection.

  STEP 4 — Verify isolation  [run during or right after attack]
  ──────────────────────────────────────────────────────────────
  mininet> h1 ping 10.0.0.10 -c 5        # ✅ must still reach server
  mininet> h_attack ping 10.0.0.10 -c 5  # ❌ must be dropped (100% loss)

  STEP 5 — Recovery  [after DROP_HARD_TIMEOUT_S = 60 s]
  ───────────────────────────────────────────────────────
  mininet> h1 iperf -c 10.0.0.10 -t 15   # throughput must recover fully

  Exit
  ────
  mininet> exit
"""
    )
    info(f"{sep}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Zone SDN ML IDS Topology")
    parser.add_argument(
        "--controller-ip",
        default=CONTROLLER_IP,
        help=f"Ryu controller IP (default: {CONTROLLER_IP})",
    )
    args = parser.parse_args()
    setLogLevel("info")
    build_and_run(controller_ip=args.controller_ip)