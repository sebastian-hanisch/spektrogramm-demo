"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """x = (1, 2, 3, 4), L = 2, Schrittweite 1, Rechteckfenster: Rahmen (1, 2), (2, 3), (3, 4) mit den Mitten 1, 2, 3.
    rfft (nfft = 2): (a + b, a - b) = (3, -1), (5, -1), (7, -1); Leistung (9, 1), (25, 1), (49, 1); Kurzzeitenergie 1 + 4 = 5, 4 + 9 = 13, 9 + 16 = 25."""
    return np.array([1.0, 2.0, 3.0, 4.0])
