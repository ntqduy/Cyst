"""Preload the local full-scale experiment package for shared runner imports."""
from __future__ import annotations

import importlib

try:
    importlib.import_module("Proposal_Model_Experiment")
except ModuleNotFoundError:
    pass
