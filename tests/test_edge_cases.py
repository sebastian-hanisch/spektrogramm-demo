"""Randfälle: alle Signaltypen, Längen, Fenster, Überlappungen, Parametergrenzen, Rauschen, Abbildungen und PDF."""

import numpy as np
import pytest

import spg_constants as C
import spg_evaluation as ev
import spg_visualization as V
from spg_pdf_export import build_pdf
from sweeps_cache import get


def settings(kind, params, **kw):
    p = list(params) + [0.0] * (3 - len(params))
    return ev.Settings(signal=kind, p1=p[0], p2=p[1], p3=p[2], **kw)


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_type_runs_for_the_shortest_and_the_longest_record_and_every_window_length(kind):
    defaults = [spec[4] for spec in C.PARAMS[kind]]
    for n in (min(C.N_CHOICES), max(C.N_CHOICES)):
        for L in (L for L in C.L_CHOICES if L <= n):
            a = ev.analyse(settings(kind, defaults, n=n, length=L, noise="an", snr=5))
            assert np.isfinite(a.power).all() and a.power.shape[0] == len(a.centers) and 0 <= a.frame_index < len(a.centers)


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_parameter_at_its_limits_runs_with_every_window_overlap_and_padding(kind):
    specs = C.PARAMS[kind]
    for pick in ("lo", "hi"):
        params = [spec[1] if pick == "lo" else spec[2] for spec in specs]
        for window in C.WINDOW_CHOICES:
            for overlap in C.OVERLAP_CHOICES:
                a = ev.analyse(settings(kind, params, n=256, length=32, window=window, overlap=overlap, pad=C.PAD_CHOICES[-1]))
                assert np.isfinite(a.power).all()
    verdict = ev.verdict_main(ev.analyse(settings(kind, [spec[2] for spec in specs], n=256, length=256)))
    assert verdict[0] in ("success", "warning", "info") and verdict[1]


def test_extreme_noise_levels_and_seed_limits_run():
    for snr in (C.SNR_MIN, C.SNR_MAX):
        for seed in (0, C.SEED_MAX):
            a = ev.analyse(settings("Chirp", (0.1, 0.3), noise="an", snr=snr, seed=seed))
            assert np.isfinite(a.rmse)


def test_power_db_is_bounded_and_survives_an_all_zero_spectrum():
    p = np.array([[1.0, 0.1, 1e-9, 0.0]])
    db = V.power_db(p)
    assert db.max() == 0.0 and db.min() == C.DB_FLOOR and db[0, 1] == pytest.approx(-10.0)
    assert V.power_db(np.zeros((2, 3))).tolist() == [[C.DB_FLOOR] * 3] * 2


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_figures_and_pdf_build_for_every_type(kind):
    a = ev.analyse(settings(kind, [spec[4] for spec in C.PARAMS[kind]], noise="an", snr=10))
    for fig in (V.build_signal(a), V.build_spectrogram(a), V.build_frame(a)):
        assert len(fig.data) >= 1
    if kind == "Zwei Impulse":
        assert len(V.build_energy(a).data) == 1
    pdf = build_pdf(a)
    assert pdf[:4] == b"%PDF" and len(pdf) > 800


def test_figures_for_rectangle_window_and_the_experiments():
    a = ev.analyse(settings("Chirp", (0.1, 0.3), window="Rechteck"))
    assert len(V.build_frame(a).layout.annotations) == 0
    assert len(V.build_resolution(get("resolution")).data) == 4 and len(V.build_chirp(get("chirp")).data) == 4 and len(V.build_pair(get("pair")).data) == 4
    assert len(V.build_correlation(get("correlation")).data) == 3 and len(V.build_threshold(get("threshold")).data) == 2
    assert len(V.build_frame(ev.analyse(settings("Chirp", (0.1, 0.3)))).layout.annotations) == 1


def test_pdf_for_every_kennzahl_branch():
    for kind, params, extra in (("Chirp", (0.1, 0.3), {}), ("Zwei parallele Chirps", (0.1, 0.25, 0.04), {}), ("Frequenzsprung", (0.2, 0.3, 0.5), {}), ("Zwei Impulse", (64.0, 0.5), {"length": 64}),
                                ("Zwei Impulse", (64.0, 0.5), {"length": 256, "window": "Rechteck"})):
        assert build_pdf(ev.analyse(settings(kind, params, **extra)))[:4] == b"%PDF"
