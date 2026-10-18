"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

from pathlib import Path

import numpy as np
import pytest

import signal_generator as G
import spg_constants as C
import spg_evaluation as ev
import spg_resolution as R
import spg_stft as T
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def preset_settings(name, **over):
    p = C.PRESETS[name]
    base = dict(n=p["n"], signal=p["signal"], p1=p["params"][0], p2=p["params"][1], p3=p["params"][2], length=p["length"], window=p["window"], overlap=p["overlap"], pad=p["pad"], frame=p["frame"],
                noise=p["noise"], snr=p["snr"], seed=p["seed"])
    return ev.Settings(**{**base, **over})


def A(name, **over):
    return ev.analyse(preset_settings(name, **over))


def de(x, k=4):
    return f"{x:.{k}f}".replace(".", ",")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_fitting_window_follows_the_chirp_to_two_ten_thousandths():
    name = "Chirp mit passender Fensterlänge"
    a = A(name)
    assert a.settings.length == 128 and a.settings.snr == 10 and de(a.rmse) == "0,0002" and "0,0002" in C.PRESET_HELP[name]


def test_preset_short_window_at_equal_noise_is_twenty_to_one_hundred_times_worse():
    rows = {(r["snr"], r["L"]): r["rmse"] for r in get("chirp")}
    assert 90 < rows[(0, 16)] / rows[(0, 128)] < 110 and 15 < rows[(10, 16)] / rows[(10, 128)] < 25
    assert "etwa das 20- bis 100-Fache" in C.PRESET_HELP["Zu kurzes Fenster: die Frequenz verschwimmt"]


def test_preset_short_window_is_over_three_hundred_times_worse_and_long_window_about_nineteen_times():
    good = A("Chirp mit passender Fensterlänge").rmse
    short = A("Zu kurzes Fenster: die Frequenz verschwimmt")
    long = A("Zu langes Fenster: der Chirp verschmiert")
    assert short.settings.length == 16 and short.settings.snr == 0 and short.rmse / good > 300 and "Dreihundertfache" in C.PRESET_HELP["Zu kurzes Fenster: die Frequenz verschwimmt"]
    assert long.settings.length == 512 and long.settings.snr == 10 and 18.5 < long.rmse / good < 19.5 and "19-Fache" in C.PRESET_HELP["Zu langes Fenster: der Chirp verschmiert"]


def test_preset_parallel_chirps_only_the_middle_lengths_separate_them():
    name = "Zwei parallele Chirps: nur ein Fenster trennt sie"
    shares = {L: A(name, length=L).both_share for L in (32, 128, 512)}
    assert A(name).settings.p3 == 0.02 and shares[128] == 1.0 and shares[32] < 0.1 and shares[512] == 0.0
    assert "Abstand 0,02" in C.PRESET_HELP[name] and "Fensterlänge 128" in C.PRESET_HELP[name] and "bei 32" in C.PRESET_HELP[name] and "bei 512" in C.PRESET_HELP[name]


def test_preset_two_impulses_long_window_sees_one_and_length_sixty_four_sees_two():
    name = "Zwei Impulse: das lange Fenster sieht nur einen"
    a = A(name)
    assert a.settings.length == 256 and a.settings.p1 == 64.0 and a.resolved is False and a.limit_time == 129.0
    assert A(name, length=64).resolved is True
    assert "Abstand 64" in C.PRESET_HELP[name] and "L/2 = 128" in C.PRESET_HELP[name] and "Länge 64" in C.PRESET_HELP[name]


# --- Auflösung --------------------------------------------------------------------------------------------------------------------------------


def test_tones_are_separated_from_two_and_a_half_bins_for_every_length():
    freq = {(r["L"], r["delta"]): r["share"] for r in get("resolution")["freq"]}
    for L in (32, 64, 128, 256):
        assert [freq[(L, d)] for d in (2.5, 3.0, 3.5, 4.0)] == [1.0] * 4 and freq[(L, 2.0)] < 0.5
        for d, ref in ((1.0, 0.175), (1.5, 0.26), (2.0, 0.485)):
            assert freq[(L, d)] == pytest.approx(ref, abs=0.006)
    assert C.FREQ_LIMIT_BINS == 2.5 and "17,5 / 26 / 48,5 / 100 %" in README
    assert abs(100 * (0.25 / C.EXP_RES_RUNS) ** 0.5 - 3.5) < 0.05 and "Standardfehler bis 3,5 Prozentpunkte" in README


def phase_grid_share(L, delta, window="Hann", m=720):
    """Anteil getrennter Paare über m gleichmäßig verteilte Phasendifferenzen (Ton bei 0,2, Nullen bis 8 L): der Zufall der Stichprobe fällt weg."""
    t = np.arange(L)
    w = T.window_values(window, L)
    ok = 0
    for k in range(m):
        x = np.cos(2 * np.pi * 0.2 * t) + np.cos(2 * np.pi * (0.2 + delta / L) * t + 2 * np.pi * k / m)
        ok += T.resolved(np.abs(np.fft.rfft(x * w, 8 * L)) ** 2, int(round(0.2 * 8 * L)), int(round((0.2 + delta / L) * 8 * L)))
    return ok / m


def test_phase_grid_shares_in_the_readme():
    assert [de(100 * phase_grid_share(32, d), 1) for d in (1.0, 1.5, 2.0, 2.5)] == ["19,6", "31,9", "49,9", "100,0"] and "19,6 / 31,9 / 49,9 / 100 % bei L = 32" in README
    assert (de(100 * phase_grid_share(32, 1.5, "Rechteck"), 1), de(100 * phase_grid_share(64, 1.5, "Rechteck"), 1)) == ("87,2", "85,4") and "87,2 % bei L = 32, 85,4 % bei L = 64" in README
    assert phase_grid_share(64, 2.5) == 1.0


def test_rectangle_window_separates_tones_earlier_but_impulses_later():
    share = R.two_tone_resolved_share(32, 1.5, G.SplitMix64(5), 100, window="Rechteck")
    assert share == 0.88 and "bei 1,5 Bins in 88 % der Läufe" in README
    assert [R.min_resolved_distance(L, "Rechteck") for L in (16, 32)] == [17, 33] and "erst ab L + 1" in README


def test_impulses_are_separated_from_half_the_window_plus_one_and_the_product_is_one_and_a_quarter():
    time = {r["L"]: r["dt"] for r in get("resolution")["time"]}
    assert time == {16: 9, 32: 17, 64: 33, 128: 65} and "9 / 17 / 33 / 65" in README
    assert C.FREQ_LIMIT_BINS * 0.5 == 1.25 and "ist etwa 1,25 (genauer 1,25 + 2,5/L)" in README and "1,25" in ev.verdict_resolution(get("resolution"))[1]


# --- Chirp ------------------------------------------------------------------------------------------------------------------------------------


def test_chirp_best_length_is_128_at_every_snr_and_the_table_numbers():
    rows = {(r["snr"], r["L"]): r["rmse"] for r in get("chirp")}
    for snr in C.EXP_CHIRP_SNR:
        assert min(C.EXP_CHIRP_LS, key=lambda L: rows[(snr, L)]) == 128
    assert max(rows[(60, L)] for L in C.EXP_CHIRP_LS) <= 0.0003
    for snr, line in ((0, "0,0740 / 0,0336 / 0,0078 / 0,0007 / 0,0038 / 0,0081"), (10, "0,0042 / 0,0015 / 0,0005 / 0,0002 / 0,0013 / 0,0047")):
        assert " / ".join(de(rows[(snr, L)]) for L in C.EXP_CHIRP_LS) == line and line in README
    assert (C.EXP_CHIRP_F0, C.EXP_CHIRP_F1, C.EXP_CHIRP_N, C.EXP_CHIRP_RUNS) == (0.1, 0.3, 1024, 100) and "von 0 bis 60 dB" in README


def test_parallel_chirps_shares_in_the_readme():
    share = {(r["df"], r["L"]): r["share"] for r in get("pair")}
    assert " / ".join(f"{round(100 * share[(0.02, L)]):d}" for L in C.EXP_PAIR_LS) == "0 / 23 / 100 / 85 / 0" and "0 / 23 / 100 / 85 / 0 %" in README
    assert " / ".join(f"{round(100 * share[(0.04, L)]):d}" for L in C.EXP_PAIR_LS) == "32 / 100 / 100 / 100 / 100" and "32 / 100 / 100 / 100 / 100 %" in README
    assert round(100 * share[(0.06, 32)]) == 88 and all(share[(0.06, L)] == 1.0 for L in C.EXP_PAIR_LS[1:]) and "bei df = 0,06 trennt L = 32 in 88 %" in README
    assert all(share[(0.10, L)] == 1.0 for L in C.EXP_PAIR_LS) and "bei df = 0,10 trennen alle Längen in 100 %" in README


# --- Korrelation und Schwelle -----------------------------------------------------------------------------------------------------------------


def test_correlation_numbers_in_the_readme_and_the_app():
    rows = {r["hop"]: r for r in get("correlation")}
    assert [de(rows[h]["frames"], 3) for h in (64, 32, 16, 8)] == ["0,002", "0,028", "0,436", "0,815"] and "0,002 / 0,028 / 0,436 / 0,815" in README
    assert all(abs(rows[h]["bin1"] - 4 / 9) < 0.01 and abs(rows[h]["bin2"] - 0.03) < 0.01 for h in rows) and "0,44 (das Quadrat von 2/3), Bins im Abstand 2: 0,03" in README
    assert "(2/3)^2 \\approx 0{,}44" in APP and "bis über 0,8" in APP and rows[8]["frames"] > 0.8 and "Bins 0,44" in APP


def test_threshold_numbers_in_the_readme():
    rows = {r["hop"]: r for r in get("threshold")}
    assert [rows[h]["cells"] for h in (64, 32, 16)] == [496, 961, 1891] and "496 / 961 / 1891 Zellen" in README
    assert " / ".join(de(100 * rows[h]["alarm_all"], 1) for h in (64, 32, 16)) == "3,5 / 4,0 / 4,5" and "gibt 3,5 / 4,0 / 4,5 % Falschalarm" in README
    assert " / ".join(de(100 * rows[h]["alarm_independent"], 1) for h in (64, 32, 16)) == "3,5 / 8,4 / 15,4" and "gibt sie 3,5 / 8,4 / 15,4 %" in README
    assert all(rows[h]["alarm_all"] < 0.05 for h in rows) and rows[16]["alarm_independent"] > 0.15


# --- Konstanten, Text der App und README ------------------------------------------------------------------------------------------------------


def test_method_numbers_in_the_readme_match_the_constants():
    assert (min(C.N_CHOICES), max(C.N_CHOICES), min(C.L_CHOICES), max(C.L_CHOICES)) == (256, 2048, 16, 512) and "n = 256 bis 2048, L = 16 bis 512" in README
    assert C.EXP_SEED == 940000 and "Seed 940000" in README and (C.EXP_RES_RUNS, C.EXP_CHIRP_RUNS, C.EXP_CORR_RUNS, C.EXP_THR_RUNS) == (200, 100, 4000, 3000)
    assert "200 Läufe mit zufälligen Phasen bei der Auflösung, 100 beim Chirp, 4000 bei der Korrelation und 3000 bei der Schwelle" in README
    assert (C.EXP_CORR_L, C.EXP_CORR_N, C.EXP_THR_L, C.EXP_THR_N, C.EXP_THR_ALPHA) == (64, 512, 64, 1024, 0.05) and max(C.PAD_CHOICES) == 8 and min(C.PAD_CHOICES) == 1


def test_hann_limits_in_the_app_text_and_the_metrics():
    a = ev.analyse(ev.Settings())
    assert a.limit_freq == pytest.approx(2.5 / 128) and a.limit_time == 65.0
    assert "2,5 Bins" in APP and "L/2" in APP and "−60" in APP and C.DB_FLOOR == -60.0 and "4 Bins" in APP
    assert "wenige Sekunden" in README


def test_hand_set_mini_instance_numbers_in_the_readme():
    assert "(9, 1), (25, 1), (49, 1)" in README and "5, 13, 25" in README


def test_the_signals_and_the_thresholds_used_in_the_texts():
    assert C.OK_RMSE == 0.005 and C.OK_SHARE == 0.9 and C.PAIR_TOLERANCE == 0.25 and "Viertels des Abstands" in ev.verdict_main(ev.analyse(ev.Settings(signal="Zwei parallele Chirps", p1=0.1, p2=0.25, p3=0.04)))[1]
