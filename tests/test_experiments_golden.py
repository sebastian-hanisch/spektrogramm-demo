"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

FREQ = {(32, 1.0): 0.175, (32, 1.5): 0.26, (32, 2.0): 0.485, (32, 2.5): 1.0, (32, 3.0): 1.0, (32, 3.5): 1.0, (32, 4.0): 1.0, (64, 1.0): 0.175, (64, 1.5): 0.255, (64, 2.0): 0.485, (64, 2.5): 1.0, (64, 3.0): 1.0, (64, 3.5): 1.0, (64, 4.0): 1.0, (128, 1.0): 0.175, (128, 1.5): 0.26, (128, 2.0): 0.485, (128, 2.5): 1.0, (128, 3.0): 1.0, (128, 3.5): 1.0, (128, 4.0): 1.0, (256, 1.0): 0.175, (256, 1.5): 0.255, (256, 2.0): 0.485, (256, 2.5): 1.0, (256, 3.0): 1.0, (256, 3.5): 1.0, (256, 4.0): 1.0}
TIME = {16: 9, 32: 17, 64: 33, 128: 65}
CHIRP = {(60, 16): 0.000208, (60, 32): 1.5e-05, (60, 64): 2e-06, (60, 128): 1e-06, (60, 256): 4e-06, (60, 512): 5.1e-05, (20, 16): 0.001327, (20, 32): 0.000471, (20, 64): 0.000171, (20, 128): 7.4e-05, (20, 256): 0.000413, (20, 512): 0.002598, (10, 16): 0.004229, (10, 32): 0.001494, (10, 64): 0.000541, (10, 128): 0.000234, (10, 256): 0.001316, (10, 512): 0.004732, (0, 16): 0.074045, (0, 32): 0.033629, (0, 64): 0.007842, (0, 128): 0.000749, (0, 256): 0.00377, (0, 512): 0.008125}
PAIR = {(0.02, 32): 0.0, (0.02, 64): 0.229508, (0.02, 128): 1.0, (0.02, 256): 0.846154, (0.02, 512): 0.0, (0.04, 32): 0.32, (0.04, 64): 1.0, (0.04, 128): 1.0, (0.04, 256): 1.0, (0.04, 512): 1.0, (0.06, 32): 0.88, (0.06, 64): 1.0, (0.06, 128): 1.0, (0.06, 256): 1.0, (0.06, 512): 1.0, (0.1, 32): 1.0, (0.1, 64): 1.0, (0.1, 128): 1.0, (0.1, 256): 1.0, (0.1, 512): 1.0}
CORR = {64: (0.001963, 0.440711, 0.032286), 32: (0.027857, 0.439982, 0.02945), 16: (0.436006, 0.442898, 0.034285), 8: (0.814561, 0.442584, 0.033426)}
THR = {64: (496, 9.176823, 0.034667, 9.176823, 0.034667), 32: (961, 9.838196, 0.04, 9.176823, 0.083667), 16: (1891, 10.51507, 0.045333, 9.176823, 0.154333)}


def test_resolution_golden():
    res = get("resolution")
    assert len(res["freq"]) == len(FREQ) == 28 and len(res["time"]) == len(TIME) == 4
    for x in res["freq"]:
        assert x["share"] == pytest.approx(FREQ[(x["L"], x["delta"])], abs=1e-6) and x["runs"] == 200
    for x in res["time"]:
        assert x["dt"] == TIME[x["L"]] and x["ratio"] == pytest.approx(TIME[x["L"]] / x["L"])


def test_chirp_golden():
    rows = get("chirp")
    assert len(rows) == len(CHIRP) == 24
    for x in rows:
        assert x["rmse"] == pytest.approx(CHIRP[(x["snr"], x["L"])], abs=1e-6) and x["runs"] == 100


def test_pair_golden():
    rows = get("pair")
    assert len(rows) == len(PAIR) == 20
    for x in rows:
        assert x["share"] == pytest.approx(PAIR[(x["df"], x["L"])], abs=1e-6)


def test_correlation_golden():
    rows = get("correlation")
    assert [x["hop"] for x in rows] == list(CORR)
    for x in rows:
        assert (x["frames"], x["bin1"], x["bin2"]) == pytest.approx(CORR[x["hop"]], abs=1e-6)


def test_threshold_golden():
    rows = get("threshold")
    assert [x["hop"] for x in rows] == list(THR)
    for x in rows:
        assert x["cells"] == THR[x["hop"]][0] and x["runs"] == 3000 and x["independent"] == 496
        assert (x["z_all"], x["alarm_all"], x["z_independent"], x["alarm_independent"]) == pytest.approx(THR[x["hop"]][1:], abs=1e-6)
