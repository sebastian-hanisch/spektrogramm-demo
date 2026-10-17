"""Testsignale: Handrechnung, Ableitung der Phase, Spuren, Picklebarkeit (st.cache_data braucht das)."""

import math
import pickle

import numpy as np
import pytest

import spg_signals as S

TWO_PI = 2 * math.pi


def test_linear_chirp_by_hand():
    s = S.linear_chirp(5, 0.0, 0.5)                      # Phase 2 pi t^2 / 16
    t = np.arange(5)
    assert s.x == pytest.approx(np.cos(TWO_PI * t ** 2 / 16))
    assert s.x[0] == 1.0 and s.x[2] == pytest.approx(0.0, abs=1e-12) and s.x[4] == pytest.approx(1.0)
    assert s.tracks[0](np.array([0, 4])).tolist() == [0.0, 0.5] and s.events == ()


def test_chirp_track_is_the_derivative_of_the_phase_at_half_times():
    n, f0, f1 = 300, 0.1, 0.3
    s = S.linear_chirp(n, f0, f1, 0.7)
    t = np.arange(n)
    phase = np.unwrap(np.arctan2(np.sin(TWO_PI * (f0 * t + (f1 - f0) * t * t / (2 * (n - 1))) + 0.7), np.cos(TWO_PI * (f0 * t + (f1 - f0) * t * t / (2 * (n - 1))) + 0.7)))
    assert np.diff(phase) / TWO_PI == pytest.approx(s.tracks[0](t[:-1] + 0.5), abs=1e-9)
    assert s.x == pytest.approx(np.cos(phase))


def test_parallel_chirps_are_a_sum_with_constant_distance():
    n = 200
    s = S.parallel_chirps(n, 0.1, 0.25, 0.04, 0.0)
    t = np.arange(n)
    a, b = s.tracks
    assert (b(t) - a(t)).tolist() == pytest.approx([0.04] * n) and a(np.array([0, n - 1])).tolist() == pytest.approx([0.1, 0.25])
    first = np.cos(TWO_PI * (0.1 * t + 0.15 * t * t / (2 * (n - 1))))
    second = np.cos(TWO_PI * (0.14 * t + 0.15 * t * t / (2 * (n - 1))))
    assert s.x == pytest.approx(first + second)
    assert S.parallel_chirps(n, 0.1, 0.25, 0.04, 1.0).x[0] == pytest.approx(1 + math.cos(1.0))


def test_frequency_step_by_hand():
    s = S.frequency_step(10, 0.25, 0.1, 4, 0.5)
    t = np.arange(10)
    assert s.x[:4] == pytest.approx(np.cos(TWO_PI * 0.25 * t[:4])) and s.x[4:] == pytest.approx(np.cos(TWO_PI * 0.1 * t[4:] + 0.5))
    assert s.tracks[0](t).tolist() == [0.25] * 4 + [0.1] * 6 and s.events == (4.0,)
    assert float(s.tracks[0](3.9)) == 0.25 and float(s.tracks[0](4.0)) == 0.1


def test_two_impulses_by_hand():
    s = S.two_impulses(20, 5, 7)
    assert np.flatnonzero(s.x).tolist() == [5, 12] and s.x.sum() == 2.0 and s.events == (5.0, 12.0) and s.tracks == ()


def test_tone_by_hand():
    s = S.tone(8, 0.25, 0.0)
    assert s.x == pytest.approx([1, 0, -1, 0, 1, 0, -1, 0], abs=1e-12)
    assert s.tracks[0](np.arange(3)).tolist() == [0.25, 0.25, 0.25] and float(s.tracks[0](5.0)) == 0.25
    assert S.tone(8, 0.25, math.pi / 2).x[0] == pytest.approx(0.0, abs=1e-12)


def test_every_signal_is_picklable():
    for s in (S.linear_chirp(64, 0.1, 0.3), S.parallel_chirps(64, 0.1, 0.2, 0.03), S.frequency_step(64, 0.2, 0.3, 30), S.two_impulses(64, 10, 5), S.tone(64, 0.2)):
        r = pickle.loads(pickle.dumps(s))
        assert r.x.tolist() == s.x.tolist() and r.events == s.events and len(r.tracks) == len(s.tracks)
        for a, b in zip(r.tracks, s.tracks):
            assert a(np.arange(10)).tolist() == b(np.arange(10)).tolist()


def test_tone_phase_is_added():
    assert S.tone(4, 0.25, 1.0).x == pytest.approx([math.cos(1.0), math.cos(math.pi / 2 + 1.0), math.cos(math.pi + 1.0), math.cos(1.5 * math.pi + 1.0)])
