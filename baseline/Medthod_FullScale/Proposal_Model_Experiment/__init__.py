"""Compatibility alias for runner imports.

The shared training runner imports the experiment package as
``Proposal_Model_Experiment``.  When ``baseline/Medthod_FullScale`` is placed
first on PYTHONPATH, this package redirects that import to the files in this
folder without changing the global runner.
"""
from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1])]
