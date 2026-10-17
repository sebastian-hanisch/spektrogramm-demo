"""Auflösung: zwei Töne in der Frequenz, zwei Impulse in der Zeit, Korrelation der Zellen. Reine Funktionen; Zufall nur über einen übergebenen SplitMix64."""

import math

import numpy as np

import spg_stft as T

TWO_PI = 2.0 * math.pi


def two_tone_resolved_share(length: int, delta_bins: float, rng, runs: int, f1: float = 0.2, pad: int = 8, window: str = "Hann") -> float:
    """Anteil der Läufe (zufällige Phasen), in denen zwei gleich starke Töne im Abstand delta_bins (in Bins 1/L) im Spektrum eines Rahmens der Länge L getrennt sind (Einbruch unter 50 %)."""
    t = np.arange(length)
    f2 = f1 + delta_bins / length
    w = T.window_values(window, length)
    ok = 0
    for _ in range(runs):
        ph = TWO_PI * rng.uniform(2)
        x = np.cos(TWO_PI * f1 * t + ph[0]) + np.cos(TWO_PI * f2 * t + ph[1])
        spec = np.abs(np.fft.rfft(x * w, pad * length)) ** 2
        ok += T.resolved(spec, int(round(f1 * pad * length)), int(round(f2 * pad * length)))
    return ok / runs


def two_impulses_resolved(length: int, dt: int, window: str = "Hann") -> bool:
    """Zwei Einzelimpulse im Abstand dt: in der Kurzzeitenergie (Schrittweite 1, Fenster der Länge L) getrennt, wenn der Einbruch unter 50 % liegt."""
    n = 4 * length
    x = np.zeros(n)
    t1 = n // 2
    x[t1] = 1.0
    x[t1 + dt] = 1.0
    energy = T.short_time_energy(x, length, window)
    return T.resolved(energy, t1 - length // 2, t1 + dt - length // 2)


def min_resolved_distance(length: int, window: str = "Hann") -> int | None:
    """Kleinster Abstand dt (1 .. 2 L - 1), bei dem zwei Impulse in der Kurzzeitenergie getrennt sind; None, wenn keiner."""
    for dt in range(1, 2 * length):
        if two_impulses_resolved(length, dt, window):
            return dt
    return None


def correlation(a, b) -> float:
    """Korrelationskoeffizient zweier gleich langer Felder (flach)."""
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
