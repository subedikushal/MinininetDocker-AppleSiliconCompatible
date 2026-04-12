#!/usr/bin/env python3
"""
advanced_topology.py — 3-Zone SDN Topology for ML Intrusion Detection
CS 8027: Advanced Networking Architecture

ROOT-CAUSE FIX (pingall 73% drop) — carried forward from v2
─────────────────────────────────────────────────────────────
Previous version used three separate /24 subnets (10.0.1.x, 10.0.2.x,
10.0.3.x).  Mininet's SDN controller installs *L2* (MAC-based) flow rules
— it does not act as an IP router.  Cross-subnet traffic was silently
dropped because no gateway existed, even though L2 connectivity was fine.

Fix: single flat subnet 10.0.0.0/24 for all hosts.  Zone separation is
preserved at the *switch level* (s1=user, s2=server, s3=attacker), which is
all the ML controller needs to observe the correct traffic patterns.

BUG FIX (this revision)
─────────────────────────
_fix_htb_quantum() previously called `net.values()`.  Mininet does NOT
expose a `.values()` method — it provides `__iter__` (yields node *names*)
and `__getitem__`.  Calling `.values()` raises AttributeError immediately,
preventing HTB quantum correction and producing a traceback.
Fix: iterate over `net.hosts + net.switches` which returns the actual node
objects.

IP map
──────
  h1        10.0.0.1   (user zone,     s1)
  h2        10.0.0.2   (user zone,     s1)
  h3        10.0.0.3   (user zone,     s1)
  web_srv   10.0.0.10  (server zone,   s2)
  db_srv    10.0.0.11  (server zone,   s2)
  h_attack  10.0.0.20  (attacker zone, s3)

Topology diagram
────────────────
                    ┌─────────────┐
                    │  s_core(s4) │  1 Gbit/s backbone
                    └──┬────┬──┬──┘
                       │    │  │
              ┌────────┘ ┌──┘  └──────────┐
              ▼          ▼                 ▼
         ┌────────┐ ┌──────────┐  ┌────────────┐
         │ s_user │ │ s_server │  │ s_attacker │
         │  (s1)  │ │   (s2)   │  │    (s3)    │
         └─┬─┬─┬──┘ └──┬────┬──┘  └─────┬──────┘
           │ │ │       │    │            │
          h1 h2 h3  web_srv db_srv   h_attack
       .0.1 .0.2 .0.3  .0.10 .0.11     .0.20
                   (all 10.0.0.x/24)
"""

import argparse
import subprocess

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController

# ── Parameters ─────────────────────────────────────────────────────────────────
CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 6653

BW_USER = 10  # Mbit/s — user-zone access links
BW_SERVER = 100  # Mbit/s — server-zone access links
BW_ATTACKER = 100  # Mbit/s — attacker-zone access link
BW_CORE = 1000  # Mbit/s — inter-switch backbone
LINK_DELAY = "2ms"


def _fix_htb_quantum(net):
    """
    Silence `sch_htb: quantum too big` kernel warnings on high-bandwidth links.

    HTB's default quantum (= MTU, typically 1514 bytes) is flagged as too
    small for links above ~100 Mbit/s.  Setting r2q=3000 raises the effective
    quantum to 33 kB, well within HTB's accepted range for 1 Gbit/s links.

    BUG FIX: Previous version called `net.values()` which does not exist on
    Mininet objects.  Mininet exposes nodes through `net.hosts`, `net.switches`,
    and `net.controllers` — iterate over hosts + switches directly.
    """
    for node in net.hosts + net.switches:
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            try:
                out = subprocess.check_output(
                    ["tc", "qdisc", "show", "dev", intf.name],
                    stderr=subprocess.DEVNULL,
                ).decode()
                if "htb" in out:
                    subprocess.call(
                        [
                            "tc",
                            "qdisc",
                            "change",
                            "dev",
                            intf.name,
                            "root",
                            "handle",
                            "1:",
                            "htb",
                            "r2q",
                            "3000",
                        ],
                        stderr=subprocess.DEVNULL,
                    )
            except Exception:
                # Interface may not have an HTB qdisc — safe to ignore
                pass


def build_and_run(controller_ip: str = CONTROLLER_IP) -> None:
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,  # deterministic link MACs; per-host `mac=`
        # parameters set below override this for hosts
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

    # ── Link parameter dictionaries ────────────────────────────────────────────
    u = dict(bw=BW_USER, delay=LINK_DELAY, use_htb=True)
    sv = dict(bw=BW_SERVER, delay=LINK_DELAY, use_htb=True)
    at = dict(bw=BW_ATTACKER, delay=LINK_DELAY, use_htb=True)
    co = dict(bw=BW_CORE, delay=LINK_DELAY, use_htb=True)

    # Host → access switch
    net.addLink(h1, s_user, **u)
    net.addLink(h2, s_user, **u)
    net.addLink(h3, s_user, **u)
    net.addLink(web_srv, s_server, **sv)
    net.addLink(db_srv, s_server, **sv)
    net.addLink(h_attack, s_attacker, **at)

    # Access switch → core switch
    net.addLink(s_user, s_core, **co)
    net.addLink(s_server, s_core, **co)
    net.addLink(s_attacker, s_core, **co)

    # ── Start ──────────────────────────────────────────────────────────────────
    info("*** Starting network\n")
    net.start()
    _fix_htb_quantum(net)
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
  │ h1          10.0.0.1      User      s1      10  Mbit/s      │
  │ h2          10.0.0.2      User      s1      10  Mbit/s      │
  │ h3          10.0.0.3      User      s1      10  Mbit/s      │
  │ web_srv     10.0.0.10     Server    s2      100 Mbit/s      │
  │ db_srv      10.0.0.11     Server    s2      100 Mbit/s      │
  │ h_attack    10.0.0.20     Attacker  s3      100 Mbit/s      │
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
      mininet> h_attack iperf -c 10.0.0.10 -u -b 500M -t 30

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