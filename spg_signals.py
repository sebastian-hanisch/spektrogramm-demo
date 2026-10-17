"""Testsignale mit exakter Wahrheit: Chirp, zwei parallele Chirps, Frequenzsprung, zwei Impulse, Ton.

Reine Funktionen, kein Zufall. Zeit t = 0 .. n - 1, Abtastrate 1. `tracks` ist die wahre Frequenz je Komponente als Funktion der Zeit (Tupel von Funktionen), `events` die Zeitpunkte von Impulsen.
"""

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TestSignal:
    x: np.ndarray
    tracks: tuple
    events: tuple


@dataclass(frozen=True)
class LinearTrack:
    """Wahre Frequenz f0 + slope t (picklebar, anders als ein Lambda)."""
    f0: float
    slope: float

    def __call__(self, tt):
        return self.f0 + self.slope * np.asarray(tt, dtype=float)


@dataclass(frozen=True)
class StepTrack:
    """Wahre Frequenz f1 vor t0, f2 ab t0."""
    f1: float
    f2: float
    t0: int

    def __call__(self, tt):
        return np.where(np.asarray(tt, dtype=float) < self.t0, self.f1, self.f2)


@dataclass(frozen=True)
class ConstTrack:
    f: float

    def __call__(self, tt):
        return np.full(np.shape(tt), float(self.f))


def _chirp_phase(t, f0: float, f1: float, n: int) -> np.ndarray:
    return TWO_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * (n - 1)))


def linear_chirp(n: int, f0: float, f1: float, phase: float = 0.0) -> TestSignal:
    """Linearer Chirp cos(2 pi (f0 t + (f1 - f0) t^2 / (2 (n - 1))) + phase): Frequenz f0 + (f1 - f0) t / (n - 1)."""
    t = np.arange(n)
    return TestSignal(np.cos(_chirp_phase(t, f0, f1, n) + phase), (LinearTrack(f0, (f1 - f0) / (n - 1)),), ())


def parallel_chirps(n: int, f0: float, f1: float, df: float, phase: float = 0.0) -> TestSignal:
    """Zwei Chirps mit gleicher Steigung im festen Frequenzabstand df: der zweite startet bei f0 + df (Phase des zweiten verschoben um `phase`)."""
    t = np.arange(n)
    x = np.cos(_chirp_phase(t, f0, f1, n)) + np.cos(_chirp_phase(t, f0, f1, n) + TWO_PI * df * t + phase)
    slope = (f1 - f0) / (n - 1)
    return TestSignal(x, (LinearTrack(f0, slope), LinearTrack(f0 + df, slope)), ())


def frequency_step(n: int, f1: float, f2: float, t0: int, phase: float = 0.0) -> TestSignal:
    """cos(2 pi f1 t) für t < t0 und cos(2 pi f2 t + phase) ab t0 (Frequenz springt bei t0 von f1 auf f2)."""
    t = np.arange(n)
    x = np.where(t < t0, np.cos(TWO_PI * f1 * t), np.cos(TWO_PI * f2 * t + phase))
    return TestSignal(x, (StepTrack(f1, f2, t0),), (float(t0),))


def two_impulses(n: int, t1: int, dt: int) -> TestSignal:
    """Zwei Einzelimpulse der Höhe 1 bei t1 und t1 + dt (keine Frequenzspur)."""
    x = np.zeros(n)
    x[t1] = 1.0
    x[t1 + dt] = 1.0
    return TestSignal(x, (), (float(t1), float(t1 + dt)))


def tone(n: int, f: float, phase: float = 0.0) -> TestSignal:
    """Ton cos(2 pi f t + phase)."""
    t = np.arange(n)
    return TestSignal(np.cos(TWO_PI * f * t + phase), (ConstTrack(f),), ())
