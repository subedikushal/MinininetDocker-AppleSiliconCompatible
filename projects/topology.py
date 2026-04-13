#!/usr/bin/env python3
"""
advanced_topology.py — 3-Zone SDN Topology for ML Intrusion Detection
CS 8027: Advanced Networking Architecture

UNRESTRICTED BANDWIDTH EDITION
──────────────────────────────
Bandwidth limits (bw=...) and traffic shaping (use_htb=True) have been completely
removed. Links will now run as fast as the host CPU allows. This bridges the
"domain shift" gap, allowing the OpenFlow switch to report realistic flood rates
(>50,000 pps) to the Ryu controller so the ML model can detect them properly.
"""

import argparse

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController

# ── Parameters ─────────────────────────────────────────────────────────────────
CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 6653
LINK_DELAY = "2ms"


def build_and_run(controller_ip: str = CONTROLLER_IP) -> None:
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,  # deterministic link MACs; per-host `mac=`
    )

    # ── Controller ─────────────────────────────────────────────────────────────
    net.addController(
        "c0",
        controller=RemoteController,
        ip=controller_ip,
        port=CONTROLLER_PORT,
    )

    # ── Switches ───────────────────────────────────────────────────────────────
    s_user = net.addSwitch("s1", protocols="OpenFlow13")  # user zone
    s_server = net.addSwitch("s2", protocols="OpenFlow13")  # server zone
    s_attacker = net.addSwitch("s3", protocols="OpenFlow13")  # attacker zone
    s_core = net.addSwitch("s4", protocols="OpenFlow13")  # backbone

    # ── Hosts — single flat subnet 10.0.0.0/24 ────────────────────────────────
    # User zone (s1)
    h1 = net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")
    h3 = net.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03")
    # Server zone (s2)
    web_srv = net.addHost("web_srv", ip="10.0.0.10/24", mac="00:00:00:00:00:10")
    db_srv = net.addHost("db_srv", ip="10.0.0.11/24", mac="00:00:00:00:00:11")
    # Attacker zone (s3)
    h_attack = net.addHost("h_attack", ip="10.0.0.20/24", mac="00:00:00:00:00:20")

    # ── Link parameter dictionaries (NO BANDWIDTH LIMITS) ──────────────────────
    # We only apply a small delay to simulate network latency.
    link_opts = dict(delay=LINK_DELAY)

    # Host → access switch
    net.addLink(h1, s_user, **link_opts)
    net.addLink(h2, s_user, **link_opts)
    net.addLink(h3, s_user, **link_opts)
    net.addLink(web_srv, s_server, **link_opts)
    net.addLink(db_srv, s_server, **link_opts)
    net.addLink(h_attack, s_attacker, **link_opts)

    # Access switch → core switch
    net.addLink(s_user, s_core, **link_opts)
    net.addLink(s_server, s_core, **link_opts)
    net.addLink(s_attacker, s_core, **link_opts)

    # ── Start ──────────────────────────────────────────────────────────────────
    info("*** Starting network\n")
    net.start()
    _print_guide()
    CLI(net)
    net.stop()


def _print_guide() -> None:
    sep = "═" * 66
    info(f"\n{sep}\n")
    info("  3-Zone SDN ML Intrusion Detection — Experiment Guide\n")
    info(f"{sep}\n")
    info(
        """
  ┌─────────────────────────────────────────────────────────────┐
  │ Host        IP            Zone      Switch  Bandwidth       │
  ├─────────────────────────────────────────────────────────────┤
  │ h1          10.0.0.1      User      s1      Unlimited       │
  │ h2          10.0.0.2      User      s1      Unlimited       │
  │ h3          10.0.0.3      User      s1      Unlimited       │
  │ web_srv     10.0.0.10     Server    s2      Unlimited       │
  │ db_srv      10.0.0.11     Server    s2      Unlimited       │
  │ h_attack    10.0.0.20     Attacker  s3      Unlimited       │
  └─────────────────────────────────────────────────────────────┘

  STEP 1 — Verify full connectivity (expect 0% drop)
  ─────────────────────────────────────────────────
  mininet> pingall

  STEP 2 — Start iperf servers on server zone
  ───────────────────────────────────────────
  mininet> web_srv iperf -s &
  mininet> db_srv  iperf -s &

  STEP 3 — Normal traffic  [controller must NOT alert]
  ─────────────────────────────────────────────────────
  mininet> h1 iperf -c 10.0.0.10 -t 30 &
  mininet> h2 iperf -c 10.0.0.11 -t 30 &
  mininet> h3 iperf -c 10.0.0.10 -t 20 &

  STEP 4 — Attack simulation  [controller MUST alert + drop]
  ───────────────────────────────────────────────────────────
  [A] UDP flood (high-volume datagram flood)
      mininet> web_srv iperf -s -u &
      mininet> h_attack iperf -c 10.0.0.10 -u -b 1000M -t 30

  [B] SYN flood (TCP half-open exhaustion)
      mininet> h_attack hping3 -S --flood -p 80 10.0.0.10

  [C] ICMP flood
      mininet> h_attack ping -f 10.0.0.10

  STEP 5 — Verify isolation  [run DURING or right after attack]
  ──────────────────────────────────────────────────────────────
  mininet> h1 ping 10.0.0.10 -c 5       # ✅ must still pass
  mininet> h_attack ping 10.0.0.10 -c 5 # 🚫 must be blocked

  STEP 6 — Post-attack recovery  [after DROP_HARD_TIMEOUT_S=60s]
  ──────────────────────────────
  mininet> h1 iperf -c 10.0.0.10 -t 15  # bandwidth must recover

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