"""Kern: Kurzzeit-Fourier-Transformation (STFT), Kamm, Gipfel je Rahmen, Kurzzeitenergie und Šidák-Schwelle.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Abtastrate 1, Frequenzen in Zyklen je Abtastwert. Rahmen der Länge L mit Schrittweite hop, Rahmenmitte = Start + L/2, Leistung |X|^2
der rfft des gefensterten Rahmens (ohne Mittelwertabzug), optional mit Nullen auf pad mal L Punkte aufgefüllt.
"""

import math

import numpy as np

WINDOWS = ("Hann", "Rechteck")


def window_values(name: str, length: int) -> np.ndarray:
    """Periodisches Hann-Fenster 0,5 - 0,5 cos(2 pi j / L) oder Rechteckfenster der Länge L."""
    if name == "Hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)
    if name == "Rechteck":
        return np.ones(length)
    raise ValueError(f"unbekanntes Fenster: {name}")


def frame_starts(n: int, length: int, hop: int) -> np.ndarray:
    """Anfänge der vollständig in die Reihe passenden Rahmen: 0, hop, 2 hop, ... bis n - L."""
    if length < 2 or length > n or hop < 1:
        raise ValueError("Rahmenlänge außerhalb von 2 .. n oder Schrittweite unter 1")
    return np.arange(0, n - length + 1, hop)


def stft_power(x, length: int, hop: int, window: str = "Hann", pad: int = 1):
    """STFT-Leistung der Zeilen von x: (Rahmenmitten, Frequenzen, Leistung mit Form (..., Rahmen, Bins)); Bins 0 .. pad L / 2."""
    x = np.asarray(x, dtype=float)
    starts = frame_starts(x.shape[-1], length, hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, length, axis=-1)[..., ::hop, :]
    nfft = pad * length
    spec = np.fft.rfft(frames * window_values(window, length), nfft, axis=-1)
    return starts + length / 2.0, np.fft.rfftfreq(nfft), np.abs(spec) ** 2


def local_maxima(v) -> list[int]:
    """Indizes innerer Punkte, die höher als der linke und mindestens so hoch wie der rechte Nachbar sind."""
    v = np.asarray(v, dtype=float)
    return [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]]


def refine_index(spec, i: int) -> float:
    """Verfeinerter Indexort des Gipfels bei i: Scheitel der Parabel durch den Logarithmus der drei Werte (am Rand: i)."""
    if i < 1 or i > len(spec) - 2:
        return float(i)
    a, b, c = (math.log(max(float(spec[j]), 1e-300)) for j in (i - 1, i, i + 1))
    den = a - 2.0 * b + c
    return i + (0.5 * (a - c) / den if den != 0 else 0.0)


def ridge_frequencies(power) -> np.ndarray:
    """Kamm: Frequenz des höchsten Gipfels je Rahmen (Parabelverfeinerung) aus der Leistung (..., Rahmen, Bins); die Frequenzachse folgt aus der Zahl der Bins (nfft = 2 (Bins - 1))."""
    power = np.asarray(power, dtype=float)
    nfft = 2 * (power.shape[-1] - 1)
    flat = power.reshape(-1, power.shape[-1])
    out = np.array([refine_index(row, int(np.argmax(row))) / nfft for row in flat])
    return out.reshape(power.shape[:-1])


def peaks_per_frame(spec, count: int, nfft: int) -> list[float]:
    """Frequenzen der `count` höchsten lokalen Maxima eines Rahmenspektrums (Parabelverfeinerung), aufsteigend sortiert; weniger, wenn es nicht genug Maxima gibt."""
    idx = sorted(local_maxima(spec), key=lambda i: -spec[i])[:count]
    return sorted(refine_index(spec, i) / nfft for i in idx)


def short_time_energy(x, length: int, window: str = "Hann") -> np.ndarray:
    """Energie sum (w[j] x[s + j])^2 je Rahmenanfang s bei Schrittweite 1 (die Summe der Leistung über alle Frequenzen bis auf den Faktor L)."""
    x = np.asarray(x, dtype=float)
    frames = np.lib.stride_tricks.sliding_window_view(x, length)
    return np.sum((frames * window_values(window, length)) ** 2, axis=-1)


def resolved(profile, i1: int, i2: int, depth: float = 0.5) -> bool:
    """Zwei Gipfel getrennt: je ein Maximum im Fenster ± halber Abstand um i1 und i2, und das Minimum dazwischen liegt unter depth mal dem kleineren Maximum."""
    lo, hi = sorted((i1, i2))
    half = max(1, int(round((hi - lo) / 2.0)))
    profile = np.asarray(profile, dtype=float)
    a = profile[max(lo - half, 0):lo + half + 1]
    b = profile[max(hi - half, 0):hi + half + 1]
    j1 = max(lo - half, 0) + int(np.argmax(a))
    j2 = max(hi - half, 0) + int(np.argmax(b))
    if j1 >= j2:
        return False
    return bool(profile[j1:j2 + 1].min() < depth * min(profile[j1], profile[j2]))


def sidak_threshold(alpha: float, m: int) -> float:
    """Schwelle für das Maximum über m unabhängige Exp(1)-Werte (Šidák): -ln(1 - (1 - alpha)^(1 / m))."""
    return -math.log(-math.expm1(math.log1p(-alpha) / m))


def n_cells(n: int, length: int, hop: int) -> int:
    """Zahl der inneren Zellen (Rahmen mal Frequenzen 1 .. L/2 - 1 ohne Gleichanteil und Nyquist) eines Spektrogramms."""
    return len(frame_starts(n, length, hop)) * (length // 2 - 1)
