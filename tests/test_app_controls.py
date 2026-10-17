"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import spg_constants as C
from spg_presets import param_key

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SNR = "Rauschabstand (dB)"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("kind", list(C.SIGNALS))
def test_every_signal_type_runs_and_shows_its_parameter_sliders(kind):
    at = _run()
    at.selectbox(key="signal_select").set_value(kind).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    labels = _labels(at)
    assert all(spec[0] in labels for spec in C.PARAMS[kind])
    others = {spec[0] for k, specs in C.PARAMS.items() if k != kind for spec in specs} - {spec[0] for spec in C.PARAMS[kind]}
    assert not (others & set(labels))


def test_the_noise_slider_exists_only_with_noise_and_keeps_its_value():
    at = _run()
    assert SNR not in _labels(at)
    at.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert not at.exception and SNR in _labels(at)
    at2 = _run({"g": "an", "snr": "7"})
    at2.radio(key="noise_select").set_value("aus").run(timeout=TIMEOUT)
    at2.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert {x.label: x for x in at2.slider}[SNR].value == 7


def test_parameters_are_kept_per_type_when_switching():
    at = _run({"q01": "0.15", "q02": "0.35"})
    by = {x.label: x for x in at.slider}
    assert by["Startfrequenz"].value == 0.15 and by["Endfrequenz"].value == 0.35
    at.selectbox(key="signal_select").set_value("Ton").run(timeout=TIMEOUT)
    at.selectbox(key="signal_select").set_value("Chirp").run(timeout=TIMEOUT)
    by = {x.label: x for x in at.slider}
    assert by["Startfrequenz"].value == 0.15 and by["Endfrequenz"].value == 0.35


def test_the_window_length_choices_follow_the_record_length():
    at2 = _run({"n": "256", "l": "512"})
    assert at2.select_slider(key="length_select").value == 256
    at3 = _run({"n": "1024", "l": "512"})
    assert at3.select_slider(key="length_select").value == 512
    at3.select_slider(key="n_select").set_value(256).run(timeout=TIMEOUT)
    assert not at3.exception and at3.select_slider(key="length_select").value == 256


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "2048", "s": "Frequenzsprung", "q21": "0.3", "q22": "9", "q23": "0.7", "l": "64", "w": "Rechteck", "o": "50", "p": "8", "r": "0.25", "g": "an", "snr": "-99", "seed": "42"})
    assert at.select_slider(key="n_select").value == 2048 and at.selectbox(key="signal_select").value == "Frequenzsprung" and at.radio(key="noise_select").value == "an"
    assert at.select_slider(key="length_select").value == 64 and at.radio(key="window_select").value == "Rechteck" and at.select_slider(key="overlap_select").value == 50
    assert at.select_slider(key="pad_select").value == 8 and at.number_input(key="seed_input").value == 42
    by = {x.label: x for x in at.slider}
    assert by["Frequenz 1"].value == 0.3 and by["Frequenz 2"].value == 0.48 and by["Sprungzeitpunkt (Anteil der Länge)"].value == 0.7 and by[SNR].value == C.SNR_MIN and by["Lage des Rahmens"].value == 0.25
    at2 = _run({"n": "3000", "s": "unsinn", "g": "vielleicht", "l": "7", "w": "x", "o": "51", "p": "3"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="signal_select").value == C.DEFAULT_SIGNAL and at2.radio(key="noise_select").value == C.DEFAULT_NOISE
    assert at2.select_slider(key="length_select").value == C.DEFAULT_L and at2.radio(key="window_select").value == C.DEFAULT_WINDOW and at2.select_slider(key="overlap_select").value == C.DEFAULT_OVERLAP
    assert at2.select_slider(key="pad_select").value == C.DEFAULT_PAD


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Zwei Impulse: das lange Fenster sieht nur einen").click().run(timeout=TIMEOUT)
    assert not at.exception
    by = {x.label: x for x in at.slider}
    assert at.selectbox(key="signal_select").value == "Zwei Impulse" and by["Abstand dt (Abtastwerte)"].value == 64.0 and at.select_slider(key="length_select").value == 256
    assert at.select_slider(key="overlap_select").value == 88 and at.select_slider(key="pad_select").value == 1
    at.button(key="preset_Zu kurzes Fenster: die Frequenz verschwimmt").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="signal_select").value == "Chirp" and at.radio(key="noise_select").value == "an" and {x.label: x for x in at.slider}[SNR].value == 0 and at.select_slider(key="length_select").value == 16


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
    assert param_key(0, 1) == "p01_kept"
