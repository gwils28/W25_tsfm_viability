"""Read the pre-registered protocol."""

from __future__ import annotations

from pathlib import Path

import yaml

from w25.data import ROOT

PROTOCOL = ROOT / "configs" / "protocol.yaml"
RESULTS = ROOT / "results"


def load_protocol(path: Path = PROTOCOL) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
