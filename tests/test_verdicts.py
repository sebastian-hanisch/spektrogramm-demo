"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import spg_evaluation as ev
import spg_signals as S


def make_analysis(signal="Chirp", p=(0.1, 0.3, 0.0), length=128, window="Hann", rmse=None, both=None, step=None, resolved=None, events=(), hop_overlap=75):
    z = np.zeros(1)
    s = ev.Settings(signal=signal, p1=p[0], p2=p[1], p3=p[2], length=length, window=window, overlap=hop_overlap)
    hann = window == "Hann"
    return ev.Analysis(s, S.TestSignal(z, (), tuple(events)), z, z, z, z, None, None, None, rmse, both, step, resolved, 0, 2.5 / length if hann else None, length / 2 + 1 if hann else None, 100)


def test_main_verdict_chirp_threshold_and_text():
    kind, text = ev.verdict_main(make_analysis(rmse=0.0003))
    assert kind == "success" and text == "Fensterlänge 128: Der Kamm weicht im Mittel um 0,0003 von der wahren Frequenz ab. Hann trennt zwei Töne ab 0,020 und zwei Impulse ab 65 Abtastwerten."
    assert ev.verdict_main(make_analysis(rmse=0.00499))[0] == "success" and ev.verdict_main(make_analysis(rmse=0.005))[0] == "warning"
    kind, text = ev.verdict_main(make_analysis(rmse=0.0696, length=16, window="Rechteck"))
    assert kind == "warning" and text == "Fensterlänge 16: Der Kamm weicht im Mittel um 0,0696 von der wahren Frequenz ab."
    assert ev.verdict_main(make_analysis("Ton", (0.2, 0.0, 0.0), rmse=0.0001))[0] == "success"


def test_main_verdict_pair_threshold_and_required_length():
    a = make_analysis("Zwei parallele Chirps", (0.1, 0.25, 0.02), both=1.0)
    kind, text = ev.verdict_main(a)
    assert kind == "success"
    assert text == ("Fensterlänge 128: In 100 % der Rahmen liegen beide Kämme innerhalb eines Viertels des Abstands 0,020 neben der Wahrheit. "
                    "Die Trennung braucht L ≥ 125 (2,5 / df), aber kein zu langes Fenster, in dem die Steigung beide verschmiert.")
    kind, text = ev.verdict_main(make_analysis("Zwei parallele Chirps", (0.1, 0.25, 0.04), both=0.32, length=32))
    assert kind == "warning" and "In 32 % der Rahmen" in text and "L ≥ 63 (2,5 / df)" in text
    assert ev.verdict_main(make_analysis("Zwei parallele Chirps", (0.1, 0.25, 0.04), both=0.9))[0] == "success" and ev.verdict_main(make_analysis("Zwei parallele Chirps", (0.1, 0.25, 0.04), both=0.89))[0] == "warning"


def test_main_verdict_frequency_step():
    base = dict(signal="Frequenzsprung", p=(0.2, 0.3, 0.5))
    kind, text = ev.verdict_main(make_analysis(step=0.0, **base))
    assert kind == "info"
    assert text == ("Fensterlänge 128: Der Kamm wechselt innerhalb einer Schrittweite (32) um den wahren Sprung zur zweiten Frequenz; über etwa 128 Abtastwerte enthält jeder Rahmen beide Frequenzen, "
                    "die Kante ist so breit wie das Fenster. Hann trennt zwei Töne ab 0,020 und zwei Impulse ab 65 Abtastwerten.")
    assert "wechselt 48 Abtastwerte nach dem wahren Sprung" in ev.verdict_main(make_analysis(step=48.0, **base))[1]
    assert "wechselt innerhalb einer Schrittweite (32) um den wahren Sprung" in ev.verdict_main(make_analysis(step=-32.0, **base))[1]
    assert "wechselt 64 Abtastwerte vor dem wahren Sprung" in ev.verdict_main(make_analysis(step=-64.0, **base))[1]
    kind, text = ev.verdict_main(make_analysis(step=None, **base))
    assert kind == "warning" and text.startswith("Fensterlänge 128: Der Kamm erreicht die zweite Frequenz nie.")


def test_main_verdict_two_impulses():
    base = dict(signal="Zwei Impulse", p=(64.0, 0.5, 0.0), events=(512.0, 576.0))
    kind, text = ev.verdict_main(make_analysis(length=64, resolved=True, **base))
    assert kind == "success" and text == "Fensterlänge 64: Die beiden Impulse im Abstand 64 sind in der Kurzzeitenergie getrennt. Hann trennt zwei Töne ab 0,039 und zwei Impulse ab 33 Abtastwerten."
    kind, text = ev.verdict_main(make_analysis(length=256, resolved=False, **base))
    assert kind == "warning" and text == ("Fensterlänge 256: Die beiden Impulse im Abstand 64 verschwimmen zu einem Gipfel (Hann braucht etwa L/2 + 1 = 129). "
                                          "Hann trennt zwei Töne ab 0,010 und zwei Impulse ab 129 Abtastwerten.")
    kind, text = ev.verdict_main(make_analysis(length=64, resolved=None, **base))
    assert kind == "info" and text.startswith("Fensterlänge 64: Ein Impuls liegt näher als L/2 am Rand der Aufzeichnung;")
    assert "(512.0" not in text and ev.verdict_main(make_analysis(length=64, resolved=True, window="Rechteck", **base))[1].endswith("getrennt.")


def freq_row(L, d, share):
    return {"L": L, "delta": d, "share": share, "runs": 200}


def test_verdict_resolution_text():
    rows = {"freq": [freq_row(32, 2.0, 0.485), freq_row(32, 2.5, 1.0), freq_row(64, 2.0, 0.49), freq_row(64, 2.5, 1.0)], "time": [{"L": 16, "dt": 9, "ratio": 9 / 16}, {"L": 128, "dt": 65, "ratio": 65 / 128}]}
    kind, text = ev.verdict_resolution(rows)
    assert kind == "info"
    assert text == ("Zwei gleich starke Töne trennt das Hann-Fenster ab 2,5 Bins (Anteil getrennt 100 % bei 2,5 Bins, 48 % bei 2,0), für jede Fensterlänge; zwei Impulse ab L/2 + 1 (65 bei L = 128). "
                    "Das Produkt beider Grenzen ist etwa 1,25 (2,5 mal 1/2, genauer 1,25 + 2,5/L). **Wer die Frequenz schärfer will, verliert Zeitauflösung.**")


def test_verdict_chirp_names_the_best_length_at_the_lowest_snr():
    rows = [{"snr": 10, "L": 16, "rmse": 0.0042, "runs": 1}, {"snr": 10, "L": 128, "rmse": 0.0002, "runs": 1}, {"snr": 0, "L": 16, "rmse": 0.074, "runs": 1}, {"snr": 0, "L": 64, "rmse": 0.0078, "runs": 1},
            {"snr": 0, "L": 128, "rmse": 0.0007, "runs": 1}, {"snr": 0, "L": 512, "rmse": 0.0081, "runs": 1}]
    kind, text = ev.verdict_chirp(rows)
    assert kind == "info"
    assert text == ("Bei 0 dB liegt der Kammfehler bei L = 16 bei 0,0740, bei L = 128 bei 0,0007 und bei L = 512 bei 0,0081. "
                    "**Für diesen Chirp gibt es eine beste Fensterlänge (hier 128); ein kurzes Fenster löst die Frequenz grob auf, ein langes verbreitert den Gipfel (die Steigung), und dann versetzt das Rauschen den Kamm stärker.**")


def test_verdict_pair_uses_the_smallest_distance():
    rows = [{"df": 0.1, "L": L, "share": 1.0} for L in (32, 128, 512)] + [{"df": 0.02, "L": 32, "share": 0.0}, {"df": 0.02, "L": 128, "share": 1.0}, {"df": 0.02, "L": 512, "share": 0.0}]
    kind, text = ev.verdict_pair(rows)
    assert kind == "info"
    assert text == ("Bei Abstand 0,02 sind beide Chirps in 0 % der Rahmen getrennt bei L = 32, in 100 % bei der besten Länge und in 0 % bei L = 512. "
                    "**Das Spektrogramm trennt zwei Komponenten nur in einem Bereich von Fensterlängen; die Momentanfrequenz liefert nur einen Wert je Zeitpunkt (siehe Demo zur Momentanfrequenz) und trennt die beiden nicht.**")


def test_verdict_correlation_names_the_hops_in_descending_order():
    rows = [{"hop": h, "overlap": 1 - h / 64, "frames": f, "bin1": 0.44, "bin2": 0.03} for h, f in ((64, 0.002), (32, 0.028), (16, 0.436), (8, 0.815))]
    kind, text = ev.verdict_correlation(rows)
    assert kind == "info"
    assert text == ("Benachbarte Bins sind mit 0,44 korreliert; benachbarte Rahmen bei Schrittweite 32 (50 % Überlappung) mit 0,03, bei 16 (75 %) mit 0,44 und bei 8 mit 0,81. "
                    "**Mehr Überlappung bringt oberhalb von etwa 50 % kaum neue Information.**")


def test_verdict_threshold_uses_the_smallest_hop():
    rows = [{"hop": 64, "cells": 496, "independent": 496, "z_all": 9.18, "z_independent": 9.18, "alarm_all": 0.035, "alarm_independent": 0.035},
            {"hop": 16, "cells": 1891, "independent": 496, "z_all": 10.52, "z_independent": 9.18, "alarm_all": 0.0453, "alarm_independent": 0.1543}]
    kind, text = ev.verdict_threshold(rows)
    assert kind == "warning"
    assert text == ("Bei Schrittweite 16 hat das Spektrogramm 1891 Zellen. Die Šidák-Schwelle über alle Zellen (10,52) gibt 4,5 % Falschalarm, über nur 496 Zellen gezählt (9,18) 15,4 %. "
                    "**Die Schwelle hängt von der Zellzahl ab; bei korrelierten Zellen ist die wirksame Zahl kleiner als die der Zellen und nicht einfach die Zahl der Zellen bei Schrittweite L.**")
