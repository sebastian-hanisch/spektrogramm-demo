"""Auswertung: Einstellungen, Messung, Analyse je Signaltyp und die Experimentfunktionen (kleine Instanzen von Hand, feste Werte)."""

import math
import pickle

import numpy as np
import pytest

import signal_generator as G
import spg_constants as C
import spg_evaluation as ev
import spg_signals as S
import spg_stft as T


def test_hop_follows_the_overlap_and_is_at_least_one():
    assert [ev.Settings(length=128, overlap=o).hop for o in (0, 50, 75, 88)] == [128, 64, 32, 15]
    assert ev.Settings(length=16, overlap=88).hop == 2 and ev.Settings(length=2, overlap=88).hop == 1
    assert ev.Settings().has_noise is False and ev.Settings(noise="an").has_noise is True


def test_noise_sigma_by_hand_and_the_german_decimal_comma():
    assert ev.noise_sigma(0) == pytest.approx(math.sqrt(0.5)) and ev.noise_sigma(10) == pytest.approx(math.sqrt(0.05)) and ev.noise_sigma(20) == pytest.approx(math.sqrt(0.005))
    assert ev.de(0.12345, 3) == "0,123" and ev.de(2.0) == "2,0" and ev.de(-0.5, 2) == "-0,50"


def test_make_signal_for_each_type():
    n = 512
    assert ev.make_signal(ev.Settings(n=n, signal="Chirp", p1=0.1, p2=0.3)).x == pytest.approx(S.linear_chirp(n, 0.1, 0.3, 0.3).x)
    assert ev.make_signal(ev.Settings(n=n, signal="Zwei parallele Chirps", p1=0.1, p2=0.25, p3=0.04)).x == pytest.approx(S.parallel_chirps(n, 0.1, 0.25, 0.04, 0.3).x)
    st = ev.make_signal(ev.Settings(n=n, signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5))
    assert st.events == (256.0,) and st.x == pytest.approx(S.frequency_step(n, 0.2, 0.3, 256, 0.3).x)
    im = ev.make_signal(ev.Settings(n=n, signal="Zwei Impulse", p1=64.0, p2=0.5))
    assert im.events == (256.0, 320.0) and np.flatnonzero(im.x).tolist() == [256, 320]
    assert ev.make_signal(ev.Settings(n=n, signal="Ton", p1=0.2)).x == pytest.approx(S.tone(n, 0.2, 0.3).x)
    with pytest.raises(ValueError):
        ev.make_signal(ev.Settings(signal="Rechteck"))


def test_second_impulse_is_moved_inside_the_record():
    im = ev.make_signal(ev.Settings(n=512, signal="Zwei Impulse", p1=400.0, p2=0.8))
    assert im.events == (410.0, 511.0) and len(im.x) == 512 and im.x.sum() == 2.0
    assert ev.make_signal(ev.Settings(n=256, signal="Zwei Impulse", p1=2.0, p2=0.1)).events == (26.0, 28.0)


def test_measurement_adds_the_scaled_splitmix_noise_and_nothing_without_noise():
    s = ev.Settings(noise="an", snr=10, seed=5)
    sig, x = ev.build_measurement(s)
    assert x - sig.x == pytest.approx(ev.noise_sigma(10) * G.SplitMix64(5).normal(1024))
    assert (x - sig.x).std() == pytest.approx(ev.noise_sigma(10), rel=0.1)
    assert ev.build_measurement(ev.Settings(seed=5))[1].tolist() == sig.x.tolist()
    assert ev.build_measurement(ev.Settings(noise="an", snr=10, seed=6))[1].tolist() != x.tolist()


def test_analyse_chirp_and_tone_follow_the_truth():
    a = ev.analyse(ev.Settings())
    assert a.rmse < 1e-6 and len(a.centers) == 29 and a.power.shape == (29, 4 * 128 // 2 + 1) and a.cells == T.n_cells(1024, 128, 32) == 1827
    assert a.ridge.shape == (29,) and a.truth.shape == (1, 29) and a.centers[:3].tolist() == [64.0, 96.0, 128.0] and a.peaks is None and a.step_error is None and a.both_share is None and a.resolved is None
    assert a.frame_index == 14 and a.limit_freq == pytest.approx(2.5 / 128) and a.limit_time == 65.0
    t = ev.analyse(ev.Settings(signal="Ton", p1=0.2137, p2=0.0, length=64))
    assert t.rmse < 1e-5 and t.truth.tolist() == [[0.2137] * len(t.centers)]


def test_analyse_noisy_chirp_golden_and_the_pickle_roundtrip():
    s = ev.Settings(noise="an", snr=10, seed=C.DEFAULT_SEED)
    a = ev.analyse(s)
    assert a.rmse == pytest.approx(0.000197, abs=1e-6)
    b = pickle.loads(pickle.dumps(a))
    assert b.rmse == a.rmse and b.settings == s


def test_frame_index_and_limits_depend_on_the_settings():
    assert [ev.analyse(ev.Settings(frame=f)).frame_index for f in (0.0, 0.5, 1.0)] == [0, 14, 28]
    r = ev.analyse(ev.Settings(window="Rechteck"))
    assert r.limit_freq is None and r.limit_time is None and r.rmse < 1e-3


def test_analyse_frequency_step_upwards_downwards_and_without_a_second_frame():
    up = ev.analyse(ev.Settings(signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5, length=64))
    down = ev.analyse(ev.Settings(signal="Frequenzsprung", p1=0.3, p2=0.2, p3=0.5, length=64))
    assert up.step_error == 0.0 and down.step_error == 0.0 and up.rmse is None
    assert ev.analyse(ev.Settings(n=256, signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5, length=256)).step_error is None


def test_step_error_by_hand():
    s = ev.Settings(n=100, signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5)
    centers = np.array([10.0, 30.0, 50.0, 70.0])
    assert ev._step_error(np.array([0.2, 0.2, 0.3, 0.3]), centers, s) == 0.0
    assert ev._step_error(np.array([0.2, 0.2, 0.2, 0.3]), centers, s) == 20.0
    assert ev._step_error(np.array([0.2, 0.3, 0.3, 0.3]), centers, s) == -20.0
    assert ev._step_error(np.array([0.2, 0.2, 0.2, 0.2]), centers, s) is None
    s2 = ev.Settings(n=100, signal="Frequenzsprung", p1=0.3, p2=0.2, p3=0.5)
    assert ev._step_error(np.array([0.3, 0.3, 0.2, 0.2]), centers, s2) == 0.0 and ev._step_error(np.array([0.3, 0.3, 0.3, 0.3]), centers, s2) is None
    assert ev._step_error(np.array([0.3, 0.2, 0.2, 0.2]), centers, s2) == -20.0


def test_analyse_parallel_chirps_share_and_peaks():
    a = ev.analyse(ev.Settings(signal="Zwei parallele Chirps", p1=0.1, p2=0.25, p3=0.04, length=128))
    assert a.both_share == 1.0 and len(a.peaks) == 29 and all(len(p) == 2 for p in a.peaks) and a.truth.shape == (2, 29) and a.ridge is None and a.rmse is None
    bad = ev.analyse(ev.Settings(signal="Zwei parallele Chirps", p1=0.1, p2=0.25, p3=0.02, length=32))
    assert bad.both_share == pytest.approx(4 / 125) and len(bad.peaks) == 125


def test_analyse_two_impulses_resolved_unresolved_and_not_evaluable():
    assert ev.analyse(ev.Settings(signal="Zwei Impulse", p1=64.0, p2=0.5, length=64)).resolved is True
    assert ev.analyse(ev.Settings(signal="Zwei Impulse", p1=64.0, p2=0.5, length=256)).resolved is False
    assert ev.analyse(ev.Settings(n=256, signal="Zwei Impulse", p1=64.0, p2=0.1, length=256)).resolved is None
    assert ev.analyse(ev.Settings(n=512, signal="Zwei Impulse", p1=400.0, p2=0.8, length=64)).resolved is None
    a = ev.analyse(ev.Settings(signal="Zwei Impulse", p1=64.0, p2=0.5, length=64))
    assert a.rmse is None and a.both_share is None and a.step_error is None and a.ridge is None


def test_sweep_resolution_small_instance():
    r = ev.sweep_resolution(ls=(32,), deltas=(1.0, 3.0), runs=20, time_ls=(16, 32))
    assert [(x["L"], x["delta"], x["runs"]) for x in r["freq"]] == [(32, 1.0, 20), (32, 3.0, 20)] and r["freq"][1]["share"] == 1.0 and 0.0 <= r["freq"][0]["share"] < 0.5
    assert r["time"] == [{"L": 16, "dt": 9, "ratio": 9 / 16}, {"L": 32, "dt": 17, "ratio": 17 / 32}]


def test_sweep_chirp_small_instance_has_one_row_per_pair_and_the_right_error():
    rows = ev.sweep_chirp(snrs=(60, 0), ls=(32, 128), runs=5, n=512)
    assert [(r["snr"], r["L"], r["runs"]) for r in rows] == [(60, 32, 5), (60, 128, 5), (0, 32, 5), (0, 128, 5)]
    assert all(r["rmse"] >= 0 for r in rows) and rows[2]["rmse"] > 10 * rows[0]["rmse"]


def test_sweep_pair_small_instance():
    rows = ev.sweep_pair(dfs=(0.02, 0.1), ls=(32, 128), n=1024)
    assert [(r["df"], r["L"]) for r in rows] == [(0.02, 32), (0.02, 128), (0.1, 32), (0.1, 128)]
    assert [r["share"] for r in rows] == [0.0, 1.0, 1.0, 1.0]


def test_sweep_correlation_small_instance():
    rows = ev.sweep_correlation(hops=(64, 8), length=64, n=256, runs=400, k=10)
    assert [r["hop"] for r in rows] == [64, 8] and rows[0]["overlap"] == 0.0 and rows[1]["overlap"] == 0.875
    assert abs(rows[0]["frames"]) < 0.1 and rows[1]["frames"] > 0.7 and 0.3 < rows[0]["bin1"] < 0.6 and abs(rows[0]["bin2"]) < 0.15


def test_sweep_threshold_small_instance():
    rows = ev.sweep_threshold(hops=(64, 16), length=64, n=512, runs=400, alpha=0.05)
    assert rows[0]["cells"] == rows[0]["independent"] == 8 * 31 and rows[1]["cells"] == 29 * 31 and rows[1]["independent"] == 8 * 31
    assert rows[0]["z_all"] == rows[0]["z_independent"] == pytest.approx(T.sidak_threshold(0.05, 248)) and rows[1]["z_all"] > rows[1]["z_independent"]
    assert rows[1]["alarm_independent"] > rows[1]["alarm_all"] - 0.02 and rows[0]["runs"] == 400


def test_step_error_counts_only_a_strict_crossing_of_the_midpoint():
    centers = np.array([10.0, 30.0, 50.0, 70.0])
    up = ev.Settings(n=100, signal="Frequenzsprung", p1=0.2, p2=0.3, p3=0.5)
    down = ev.Settings(n=100, signal="Frequenzsprung", p1=0.3, p2=0.2, p3=0.5)
    assert ev._step_error(np.array([0.2, 0.25, 0.3, 0.3]), centers, up) == 0.0       # Wert genau auf der Mitte zählt noch nicht als Wechsel
    assert ev._step_error(np.array([0.3, 0.25, 0.2, 0.2]), centers, down) == 0.0


def test_impulse_analysis_needs_the_whole_window_around_both_impulses():
    def run(t1, dt=64):
        return ev.analyse(ev.Settings(n=512, signal="Zwei Impulse", p1=float(dt), p2=t1 / 512, length=64)).resolved
    assert run(32) is True and run(31) is None and run(33) is True        # erster Impuls: Fenster beginnt bei 0 gerade noch / nicht mehr
    assert run(417) is None and run(416) is True                          # zweiter Impuls bei 481: letztes Fenster endet bei 448 + 64 = 512; Mitte 481 - 32 = 449 = Länge der Energie


def test_step_error_without_a_step_uses_the_downward_branch():
    s = ev.Settings(n=100, signal="Frequenzsprung", p1=0.2, p2=0.2, p3=0.5)
    assert ev._step_error(np.array([0.19, 0.21, 0.21, 0.21]), np.array([10.0, 30.0, 50.0, 70.0]), s) == -40.0
