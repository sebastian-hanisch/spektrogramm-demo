"""Kern der STFT gegen Handrechnung (Mini-Instanz) und gegen scipy (nur Orakel)."""

import math

import numpy as np
import pytest
import scipy.signal as sg

import spg_stft as T


def test_mini_instance_power_centres_and_energy(mini_x):
    centres, freqs, power = T.stft_power(mini_x, 2, 1, "Rechteck")
    assert list(centres) == [1.0, 2.0, 3.0] and list(freqs) == [0.0, 0.5]
    assert power.tolist() == [[9.0, 1.0], [25.0, 1.0], [49.0, 1.0]]
    assert T.short_time_energy(mini_x, 2, "Rechteck").tolist() == [5.0, 13.0, 25.0]


def test_window_values_by_hand():
    assert T.window_values("Hann", 4).tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5])
    assert T.window_values("Hann", 2).tolist() == pytest.approx([0.0, 1.0])
    assert T.window_values("Rechteck", 3).tolist() == [1.0, 1.0, 1.0]
    assert T.WINDOWS == ("Hann", "Rechteck")
    with pytest.raises(ValueError):
        T.window_values("Blackman", 8)


def test_hann_equals_scipy_periodic_window():
    for L in (8, 33, 128):
        assert T.window_values("Hann", L) == pytest.approx(sg.get_window("hann", L))


def test_frame_starts_and_errors():
    assert T.frame_starts(10, 4, 3).tolist() == [0, 3, 6]
    assert T.frame_starts(4, 4, 1).tolist() == [0]
    assert T.frame_starts(10, 4, 100).tolist() == [0]
    for args in ((10, 11, 1), (10, 1, 1), (10, 4, 0)):
        with pytest.raises(ValueError):
            T.frame_starts(*args)
    assert T.frame_starts(10, 2, 1).tolist() == list(range(9))


@pytest.mark.parametrize("L,hop,pad", [(16, 4, 1), (64, 16, 4), (33, 7, 2)])
def test_power_against_scipy_spectrogram(L, hop, pad):
    x = np.random.default_rng(3).normal(size=300)
    centres, freqs, power = T.stft_power(x, L, hop, "Hann", pad)
    f, t, S = sg.spectrogram(x, fs=1.0, window="hann", nperseg=L, noverlap=L - hop, nfft=pad * L, detrend=False, scaling="spectrum", mode="psd")
    w = sg.get_window("hann", L)
    assert freqs == pytest.approx(f)
    assert centres == pytest.approx(t)
    scale = w.sum() ** 2
    assert power[:, [0, -1]] == pytest.approx(S.T[:, [0, -1]] * scale)          # scipy verdoppelt die inneren Bins (einseitig)
    assert power[:, 1:-1] == pytest.approx(S.T[:, 1:-1] * scale / 2.0)


def test_padding_keeps_the_values_at_the_original_bins_and_adds_bins():
    x = np.random.default_rng(4).normal(size=200)
    _, f1, p1 = T.stft_power(x, 32, 8, "Hann", 1)
    _, f4, p4 = T.stft_power(x, 32, 8, "Hann", 4)
    assert p4.shape[-1] == 4 * 16 + 1 and p1.shape[-1] == 17 and len(f4) == 65
    assert p4[:, ::4] == pytest.approx(p1)


def test_multiple_rows_have_the_leading_shape():
    x = np.random.default_rng(5).normal(size=(3, 100))
    _, _, p = T.stft_power(x, 16, 8, "Hann")
    assert p.shape == (3, 11, 9)
    _, _, p0 = T.stft_power(x[1], 16, 8, "Hann")
    assert p[1] == pytest.approx(p0)


def test_local_maxima_by_hand():
    assert T.local_maxima([0, 2, 1, 3, 3, 0]) == [1, 3]
    assert T.local_maxima([1, 1, 1]) == []
    assert T.local_maxima([5, 1, 5]) == []          # nur innere Punkte
    assert T.local_maxima([0, 3, 3, 0]) == [1]      # Plateau: der linke Rand zählt


def test_refine_index_is_exact_for_a_log_parabola_and_safe_at_edges():
    spec = np.exp(-((np.arange(10) - 3.3) ** 2))
    assert T.refine_index(spec, 3) == pytest.approx(3.3)
    assert T.refine_index(spec, 0) == 0.0 and T.refine_index(spec, 9) == 9.0
    assert T.refine_index(np.ones(5), 2) == 2.0
    assert T.refine_index(np.array([0.0, 1.0, 0.0]), 1) == pytest.approx(1.0)


def test_ridge_of_a_tone_between_bins_is_close_to_the_frequency():
    n, f = 1024, 0.2137
    x = np.cos(2 * np.pi * f * np.arange(n))
    _, _, p = T.stft_power(x, 128, 32, "Hann", 4)
    ridge = T.ridge_frequencies(p)
    assert ridge.shape == (p.shape[0],) and np.abs(ridge - f).max() < 1e-3
    two = T.ridge_frequencies(np.stack([p, p]))
    assert two.shape == (2, p.shape[0]) and two[1] == pytest.approx(ridge)


def test_peaks_per_frame_sorted_and_limited():
    nfft = 100
    spec = np.exp(-((np.arange(51) - 10.0) ** 2)) * 5 + np.exp(-((np.arange(51) - 30.4) ** 2)) * 3 + np.exp(-((np.arange(51) - 40.0) ** 2))
    p = T.peaks_per_frame(spec, 2, nfft)
    assert p == pytest.approx([0.10, 0.304], abs=1e-3)
    assert len(T.peaks_per_frame(spec, 5, nfft)) == 3 and T.peaks_per_frame(spec, 5, nfft) == sorted(T.peaks_per_frame(spec, 5, nfft))
    assert T.peaks_per_frame(np.ones(10), 2, 18) == []


def test_short_time_energy_against_brute_force():
    x = np.random.default_rng(6).normal(size=60)
    for window in T.WINDOWS:
        e = T.short_time_energy(x, 8, window)
        w = T.window_values(window, 8)
        assert len(e) == 53
        assert e == pytest.approx([float(np.sum((w * x[s:s + 8]) ** 2)) for s in range(53)])


def test_resolved_by_hand():
    assert T.resolved([0, 5, 1, 5, 0], 1, 3)
    assert not T.resolved([0, 5, 4, 5, 0], 1, 3)
    assert not T.resolved([0, 5, 0, 0, 0], 1, 2)                       # ein Gipfel, beide Fenster finden ihn
    assert T.resolved([0, 5, 4, 5, 0], 1, 3, depth=0.9) and T.resolved([0, 5, 1, 5, 0], 3, 1)
    assert T.resolved([0, 5, 2.5 - 1e-9, 5, 0], 1, 3) and not T.resolved([0, 5, 2.5, 5, 0], 1, 3)  # Einbruch genau auf 50 %: nicht getrennt
    assert T.resolved([5, 1, 5], 0, 2)                                  # Gipfel am Rand


def test_sidak_threshold_and_cell_count():
    assert T.sidak_threshold(0.05, 1) == pytest.approx(-math.log(0.05))
    for m in (2, 16, 496, 1891):
        z = T.sidak_threshold(0.05, m)
        assert (1.0 - math.exp(-z)) ** m == pytest.approx(0.95)
    assert T.sidak_threshold(0.05, 496) == pytest.approx(9.1768, abs=1e-3)
    assert T.n_cells(1024, 64, 64) == 16 * 31 and T.n_cells(1024, 64, 32) == 31 * 31 and T.n_cells(100, 10, 5) == 19 * 4
    assert T.n_cells(1024, 16, 4) == len(T.frame_starts(1024, 16, 4)) * 7


def test_local_maxima_includes_the_first_and_the_last_inner_point():
    assert T.local_maxima([0, 3, 1, 0]) == [1] and T.local_maxima([0, 1, 3, 0]) == [2]


def test_refine_index_refines_the_second_point_too():
    spec = np.exp(-((np.arange(6) - 1.3) ** 2))
    assert T.refine_index(spec, 1) == pytest.approx(1.3) and T.refine_index(spec, 4) != 4.0
