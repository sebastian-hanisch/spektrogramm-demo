"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import spg_constants as C
from spg_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "β": "beta", "π": "pi", "≥": ">=",
            "Š": "S"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Spektrogramm: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = ", ".join(f"{label} {de(v, 3)}" for (label, *_rest), v in zip(C.PARAMS[s.signal], (s.p1, s.p2, s.p3)))
    noise = f"Rauschen {s.snr} dB gegen eine Linie der Amplitude 1" if s.has_noise else "ohne Rauschen"
    pdf.multi_cell(0, 5, _s(f"{s.signal} ({params}), n = {s.n}, Fenster {s.window} der Länge {s.length}, Schrittweite {s.hop} ({s.overlap} % Überlappung), Nullen bis {s.pad} L, {noise}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Spektrogramm: {len(a.centers)} Rahmen, {a.cells} Zellen (Bins 1 bis L/2 - 1)."]
    if a.limit_freq is not None:
        out.append(f"Trenngrenze bei Hann: zwei Töne ab {de(a.limit_freq, 4)} (2,5 / L), zwei Impulse ab {de(a.limit_time, 0)} Abtastwerten (L/2 + 1).")
    if a.rmse is not None:
        out.append(f"Kammfehler (RMSE gegen die wahre Frequenz in den Rahmenmitten): {de(a.rmse, 4)}.")
    if a.both_share is not None:
        out.append(f"Rahmen mit beiden Kämmen innerhalb eines Viertels von df: {de(100 * a.both_share, 1)} %.")
    if a.step_error is not None:
        out.append(f"Wechsel des Kamms zur zweiten Frequenz: {de(a.step_error, 0)} Abtastwerte gegen den wahren Sprung.")
    if a.resolved is not None:
        out.append("Die beiden Impulse sind in der Kurzzeitenergie " + ("getrennt." if a.resolved else "nicht getrennt."))
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: spektrogramm-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
