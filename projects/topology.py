#!/usr/bin/env python3
"""
topology.py — Enterprise Multi-Tier SDN Topology for ML Intrusion Detection

Topology overview
─────────────────
                               ┌─────────┐
                               │ s_core  │  (Backbone)
                               └────┬────┘
                  ┌─────────────────┴─────────────────┐
            ┌─────┴─────┐                       ┌─────┴─────┐
            │  s_agg1   │                       │  s_agg2   │  (Aggregation)
            └─┬───────┬─┘                       └─┬───────┬─┘
      ┌───────┘       └───────┐           ┌───────┘       └───────┐
 ┌────┴────┐             ┌────┴────┐ ┌────┴────┐             ┌────┴────┐
 │s_user_a │             │s_user_b │ │s_server │             │ s_dmz   │ (Access)
 └─┬──┬──┬─┘             └───┬──┬──┘ └─┬──┬──┬─┘             └─┬──┬──┬─┘
   │  │  │                   │  │      │  │  │                 │  │  │
  h1 h2 h3                  h4  h5  web_srv db_srv api_srv   dns_srv h_attack
                                                                     h_attack2
"""

import argparse
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController

CONTROLLER_IP = "127.0.0.1"
CONTROLLER_PORT = 6653
LINK_DELAY = "2ms"


def build_and_run(controller_ip=CONTROLLER_IP):
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # ── Controller ────────────────────────────────────────────────────────────
    net.addController(
        "c0", controller=RemoteController, ip=controller_ip, port=CONTROLLER_PORT
    )

    # ── Switches (Multi-Tier) ─────────────────────────────────────────────────
    s_core = net.addSwitch("s100", protocols="OpenFlow13")

    s_agg1 = net.addSwitch("s200", protocols="OpenFlow13")
    s_agg2 = net.addSwitch("s201", protocols="OpenFlow13")

    s_user_a = net.addSwitch("s1", protocols="OpenFlow13")
    s_user_b = net.addSwitch("s2", protocols="OpenFlow13")
    s_server = net.addSwitch("s3", protocols="OpenFlow13")
    s_dmz = net.addSwitch("s4", protocols="OpenFlow13")

    # ── Hosts ─────────────────────────────────────────────────────────────────
    h1 = net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")
    h3 = net.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03")
    h4 = net.addHost("h4", ip="10.0.0.4/24", mac="00:00:00:00:00:04")
    h5 = net.addHost("h5", ip="10.0.0.5/24", mac="00:00:00:00:00:05")

    web_srv = net.addHost("web_srv", ip="10.0.0.10/24", mac="00:00:00:00:00:10")
    db_srv = net.addHost("db_srv", ip="10.0.0.11/24", mac="00:00:00:00:00:11")
    api_srv = net.addHost("api_srv", ip="10.0.0.12/24", mac="00:00:00:00:00:12")

    dns_srv = net.addHost("dns_srv", ip="10.0.0.15/24", mac="00:00:00:00:00:15")
    h_attack = net.addHost("h_attack", ip="10.0.0.20/24", mac="00:00:00:00:00:20")
    h_attack2 = net.addHost("h_attack2", ip="10.0.0.21/24", mac="00:00:00:00:00:21")

    # ── Links ─────────────────────────────────────────────────────────────────
    lk = dict(delay=LINK_DELAY)

    # Core to Aggregation
    net.addLink(s_core, s_agg1, **lk)
    net.addLink(s_core, s_agg2, **lk)

    # Aggregation to Access
    net.addLink(s_agg1, s_user_a, **lk)
    net.addLink(s_agg1, s_user_b, **lk)
    net.addLink(s_agg2, s_server, **lk)
    net.addLink(s_agg2, s_dmz, **lk)

    # Access to Hosts
    net.addLink(h1, s_user_a, **lk)
    net.addLink(h2, s_user_a, **lk)
    net.addLink(h3, s_user_a, **lk)

    net.addLink(h4, s_user_b, **lk)
    net.addLink(h5, s_user_b, **lk)

    net.addLink(web_srv, s_server, **lk)
    net.addLink(db_srv, s_server, **lk)
    net.addLink(api_srv, s_server, **lk)

    net.addLink(dns_srv, s_dmz, **lk)
    net.addLink(h_attack, s_dmz, **lk)
    net.addLink(h_attack2, s_dmz, **lk)

    # ── Start ─────────────────────────────────────────────────────────────────
    info("*** Starting Complex Enterprise Network\n")
    net.start()

    info("\n--- Network Started ---\n")
    info("Use standard mininet commands:\n")
    info("  mininet> pingall\n")
    info("  mininet> h_attack ping web_srv\n")
    info("-----------------------\n")

    CLI(net)
    net.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tier SDN ML IDS Topology")
    parser.add_argument("--controller-ip", default=CONTROLLER_IP)
    args = parser.parse_args()
    setLogLevel("info")
    build_and_run(controller_ip=args.controller_ip)
