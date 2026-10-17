"""Auflösung: Impulse in der Zeit, Töne in der Frequenz, Korrelation. Handrechnung und feste Werte."""

import numpy as np
import pytest

import signal_generator as G
import spg_resolution as R


@pytest.mark.parametrize("L", [16, 32, 64, 128])
def test_hann_separates_two_impulses_exactly_from_half_the_window_plus_one(L):
    assert not R.two_impulses_resolved(L, L // 2) and R.two_impulses_resolved(L, L // 2 + 1)
    assert R.min_resolved_distance(L) == L // 2 + 1


def test_rectangle_window_needs_the_whole_window_for_impulses():
    for L in (16, 32):
        assert R.min_resolved_distance(L, "Rechteck") == L + 1
        assert not R.two_impulses_resolved(L, L, "Rechteck") and R.two_impulses_resolved(L, L + 1, "Rechteck")


def test_min_resolved_distance_is_none_when_nothing_resolves():
    assert R.min_resolved_distance(2, "Rechteck") is None and R.min_resolved_distance(2) == 2


def test_two_tone_share_is_deterministic_for_the_same_stream_and_extremes_are_exact():
    a = R.two_tone_resolved_share(32, 1.0, G.SplitMix64(5), 100)
    assert a == R.two_tone_resolved_share(32, 1.0, G.SplitMix64(5), 100) and 0.0 < a < 0.5
    assert R.two_tone_resolved_share(32, 3.0, G.SplitMix64(5), 50) == 1.0 and R.two_tone_resolved_share(32, 0.1, G.SplitMix64(5), 50) == 0.0


def test_two_tone_share_golden_values_hann_and_rectangle():
    hann = [R.two_tone_resolved_share(32, d, G.SplitMix64(5), 100) for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    rect = [R.two_tone_resolved_share(32, d, G.SplitMix64(5), 100, window="Rechteck") for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    assert hann == pytest.approx([0.04, 0.17, 0.31, 0.52, 1.0, 1.0]) and rect == pytest.approx([0.16, 0.37, 0.88, 1.0, 1.0, 1.0])
    assert R.two_tone_resolved_share(64, 2.5, G.SplitMix64(7), 40, f1=0.2, pad=4) == 1.0


def test_correlation_by_hand():
    a = np.array([1.0, 2.0, 3.0])
    assert R.correlation(a, 2 * a + 1) == pytest.approx(1.0) and R.correlation(a, -a) == pytest.approx(-1.0)
    assert R.correlation(a, np.array([1.0, 3.0, 2.0])) == pytest.approx(0.5)
    assert R.correlation(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 5.0]])) == pytest.approx(np.corrcoef([1, 2, 3, 4], [1, 2, 3, 5])[0, 1])
