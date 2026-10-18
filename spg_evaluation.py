"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import signal_generator as G
import spg_constants as C
import spg_resolution as R
import spg_signals as S
import spg_stft as T


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def noise_sigma(snr_db: float) -> float:
    """Standardabweichung des Rauschens gegen eine Linie der Amplitude 1 (Leistung 1/2): sqrt(0,5 / 10^(SNR/10))."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    signal: str = C.DEFAULT_SIGNAL
    p1: float = C.PARAMS[C.DEFAULT_SIGNAL][0][4]
    p2: float = C.PARAMS[C.DEFAULT_SIGNAL][1][4]
    p3: float = 0.0
    length: int = C.DEFAULT_L
    window: str = C.DEFAULT_WINDOW
    overlap: int = C.DEFAULT_OVERLAP
    pad: int = C.DEFAULT_PAD
    frame: float = C.DEFAULT_FRAME
    noise: str = C.DEFAULT_NOISE
    snr: int = C.DEFAULT_SNR
    seed: int = C.DEFAULT_SEED

    @property
    def hop(self) -> int:
        """Schrittweite: L mal (1 - Überlappung), mindestens 1."""
        return max(1, int(round(self.length * (1.0 - self.overlap / 100.0))))

    @property
    def has_noise(self) -> bool:
        return self.noise == "an"


def make_signal(s: Settings) -> S.TestSignal:
    """Das saubere Signal des gewählten Typs (Anfangsphase 0,3)."""
    if s.signal == "Chirp":
        return S.linear_chirp(s.n, s.p1, s.p2, 0.3)
    if s.signal == "Zwei parallele Chirps":
        return S.parallel_chirps(s.n, s.p1, s.p2, s.p3, 0.3)
    if s.signal == "Frequenzsprung":
        return S.frequency_step(s.n, s.p1, s.p2, int(round(s.p3 * s.n)), 0.3)
    if s.signal == "Zwei Impulse":
        t1 = int(round(s.p2 * s.n))
        return S.two_impulses(s.n, t1, max(1, min(int(round(s.p1)), s.n - 1 - t1)))
    if s.signal == "Ton":
        return S.tone(s.n, s.p1, 0.3)
    raise ValueError(f"unbekanntes Signal: {s.signal}")


def build_measurement(s: Settings) -> tuple[S.TestSignal, np.ndarray]:
    sig = make_signal(s)
    x = sig.x.copy()
    if s.has_noise:
        x = x + noise_sigma(s.snr) * G.SplitMix64(s.seed).normal(s.n)
    return sig, x


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    signal: S.TestSignal
    x: np.ndarray
    centers: np.ndarray
    freqs: np.ndarray
    power: np.ndarray               # Frames x Bins
    ridge: np.ndarray | None        # Kamm je Rahmen (nicht bei Impulsen und zwei Chirps)
    peaks: list | None              # zwei Gipfel je Rahmen (nur bei zwei Chirps)
    truth: np.ndarray | None        # wahre Frequenzen in den Rahmenmitten (Spuren x Rahmen)
    rmse: float | None
    both_share: float | None
    step_error: float | None
    resolved: bool | None
    frame_index: int
    limit_freq: float | None        # 2,5 / L (nur Hann gemessen)
    limit_time: float | None        # L / 2 + 1 (nur Hann gemessen)
    cells: int


def _step_error(a_ridge, centers, s: Settings) -> float | None:
    """Zeitpunkt, an dem der Kamm die Mitte zwischen den beiden Frequenzen zur zweiten Frequenz hin überschreitet, minus der wahre Sprungzeitpunkt (None, wenn nie)."""
    mid = 0.5 * (s.p1 + s.p2)
    crossed = a_ridge > mid if s.p2 > s.p1 else a_ridge < mid
    if not crossed.any():
        return None
    return float(centers[int(np.argmax(crossed))] - int(round(s.p3 * s.n)))


def analyse(s: Settings) -> Analysis:
    sig, x = build_measurement(s)
    centers, freqs, power = T.stft_power(x, s.length, s.hop, s.window, s.pad)
    nfft = s.pad * s.length
    ridge = peaks = truth = rmse = both = step = res = None
    if s.signal in ("Chirp", "Ton", "Frequenzsprung"):
        ridge = T.ridge_frequencies(power)
        truth = np.array([tr(centers) for tr in sig.tracks])
        if s.signal != "Frequenzsprung":
            rmse = float(math.sqrt(np.mean((ridge - truth[0]) ** 2)))
        else:
            step = _step_error(ridge, centers, s)
    elif s.signal == "Zwei parallele Chirps":
        peaks = [T.peaks_per_frame(row, 2, nfft) for row in power]
        truth = np.array([tr(centers) for tr in sig.tracks])
        good = sum(len(p) == 2 and all(abs(f - t) < C.PAIR_TOLERANCE * s.p3 for f, t in zip(p, truth[:, i])) for i, p in enumerate(peaks))
        both = good / len(peaks)
    else:
        energy = T.short_time_energy(x, s.length, s.window)
        c1, c2 = (int(round(e)) - s.length // 2 for e in sig.events)
        res = T.resolved(energy, c1, c2) if 0 <= c1 and c2 < len(energy) else None
    index = int(round(s.frame * (len(centers) - 1)))
    hann = s.window == "Hann"
    return Analysis(s, sig, x, centers, freqs, power, ridge, peaks, truth, rmse, both, step, res, index, C.FREQ_LIMIT_BINS / s.length if hann else None, s.length / 2.0 + 1.0 if hann else None,
                    T.n_cells(s.n, s.length, s.hop))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def sweep_resolution(ls=C.EXP_RES_LS, deltas=C.EXP_RES_DELTAS, runs: int = C.EXP_RES_RUNS, time_ls=C.EXP_RES_TIME_LS) -> dict:
    """Auflösung. `freq`: Anteil der Läufe, in denen zwei gleich starke Töne im Abstand von delta Bins (1/L) mit Hann-Fenster getrennt sind (Einbruch unter 50 %; gleiche Phasenfolge für alle L);
    `time`: kleinster Abstand, bei dem zwei Impulse in der Kurzzeitenergie getrennt sind."""
    freq = [{"L": L, "delta": d, "share": R.two_tone_resolved_share(L, d, G.SplitMix64(C.EXP_SEED + int(round(10 * d))), runs), "runs": runs} for L in ls for d in deltas]
    time = []
    for L in time_ls:
        dt = R.min_resolved_distance(L)
        time.append({"L": L, "dt": dt, "ratio": dt / L})
    return {"freq": freq, "time": time}


def sweep_chirp(snrs=C.EXP_CHIRP_SNR, ls=C.EXP_CHIRP_LS, runs: int = C.EXP_CHIRP_RUNS, n: int = C.EXP_CHIRP_N) -> list[dict]:
    """Chirp in Rauschen: RMSE des Kamms (Gipfel je Rahmen mit Parabelverfeinerung, Schrittweite L/4) gegen die wahre Frequenz in der Rahmenmitte, je Rauschabstand und Fensterlänge."""
    sig = S.linear_chirp(n, C.EXP_CHIRP_F0, C.EXP_CHIRP_F1)
    rows = []
    for snr in snrs:
        for L in ls:
            rng = G.SplitMix64(C.EXP_SEED + 1000 + snr + L)
            x = sig.x[None, :] + noise_sigma(snr) * rng.normal(runs * n).reshape(runs, n)
            centers, _, power = T.stft_power(x, L, max(1, L // 4), "Hann", C.EXP_CHIRP_PAD)
            err = T.ridge_frequencies(power) - sig.tracks[0](centers)[None, :]
            rows.append({"snr": snr, "L": L, "rmse": float(math.sqrt(np.mean(err ** 2))), "runs": runs})
    return rows


def sweep_pair(dfs=C.EXP_PAIR_DFS, ls=C.EXP_PAIR_LS, n: int = C.EXP_PAIR_N) -> list[dict]:
    """Zwei parallele Chirps (kein Rauschen): Anteil der Rahmen, in denen die zwei höchsten Gipfel beide innerhalb df/4 der wahren Frequenzen liegen (Schrittweite L/4)."""
    rows = []
    for df in dfs:
        sig = S.parallel_chirps(n, C.EXP_PAIR_F0, C.EXP_PAIR_F1, df)
        for L in ls:
            centers, _, power = T.stft_power(sig.x, L, max(1, L // 4), "Hann", C.EXP_PAIR_PAD)
            truth = np.array([tr(centers) for tr in sig.tracks])
            peaks = [T.peaks_per_frame(row, 2, C.EXP_PAIR_PAD * L) for row in power]
            good = sum(len(p) == 2 and all(abs(f - t) < C.PAIR_TOLERANCE * df for f, t in zip(p, truth[:, i])) for i, p in enumerate(peaks))
            rows.append({"df": df, "L": L, "share": good / len(peaks)})
    return rows


def sweep_correlation(hops=C.EXP_CORR_HOPS, length: int = C.EXP_CORR_L, n: int = C.EXP_CORR_N, runs: int = C.EXP_CORR_RUNS, k: int = C.EXP_CORR_BIN) -> list[dict]:
    """Weißes Rauschen: Korrelation der Zellleistungen zwischen benachbarten Rahmen (gleicher Bin), benachbarten Bins und Bins im Abstand 2, je Schrittweite."""
    x = G.SplitMix64(C.EXP_SEED + 2000).normal(runs * n).reshape(runs, n)
    rows = []
    for hop in hops:
        _, _, p = T.stft_power(x, length, hop)
        rows.append({"hop": hop, "overlap": 1.0 - hop / length, "frames": R.correlation(p[:, :-1, k], p[:, 1:, k]), "bin1": R.correlation(p[:, :, k], p[:, :, k + 1]), "bin2": R.correlation(p[:, :, k], p[:, :, k + 2])})
    return rows


def sweep_threshold(hops=C.EXP_THR_HOPS, length: int = C.EXP_THR_L, n: int = C.EXP_THR_N, runs: int = C.EXP_THR_RUNS, alpha: float = C.EXP_THR_ALPHA) -> list[dict]:
    """Weißes Rauschen: Falschalarm des höchsten Zellwerts (Quotient zur mittleren Zellleistung) mit der Šidák-Schwelle über alle Zellen gegen die über nur die Zellen bei Schrittweite L."""
    x = G.SplitMix64(C.EXP_SEED + 3000).normal(runs * n).reshape(runs, n)
    indep = T.n_cells(n, length, length)
    rows = []
    for hop in hops:
        _, _, p = T.stft_power(x, length, hop)
        inner = p[..., 1:length // 2]
        top = (inner / inner.mean(axis=(1, 2), keepdims=True)).max(axis=(1, 2))
        cells = T.n_cells(n, length, hop)
        z_all, z_ind = T.sidak_threshold(alpha, cells), T.sidak_threshold(alpha, indep)
        rows.append({"hop": hop, "cells": cells, "independent": indep, "z_all": z_all, "z_independent": z_ind, "alarm_all": float(np.mean(top > z_all)), "alarm_independent": float(np.mean(top > z_ind)), "runs": runs})
    return rows


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    limits = f" Hann trennt zwei Töne ab {de(a.limit_freq, 3)} und zwei Impulse ab {de(a.limit_time, 0)} Abtastwerten." if a.limit_freq is not None else ""
    if s.signal in ("Chirp", "Ton"):
        ok = a.rmse < C.OK_RMSE
        return ("success" if ok else "warning"), f"Fensterlänge {s.length}: Der Kamm weicht im Mittel um {de(a.rmse, 4)} von der wahren Frequenz ab." + limits
    if s.signal == "Zwei parallele Chirps":
        ok = a.both_share >= C.OK_SHARE
        need = math.ceil(round(C.FREQ_LIMIT_BINS / s.p3, 6))
        return ("success" if ok else "warning"), (f"Fensterlänge {s.length}: In {de(100 * a.both_share, 0)} % der Rahmen liegen beide Kämme innerhalb eines Viertels des Abstands {de(s.p3, 3)} neben der Wahrheit. "
                                                  f"Die Trennung braucht L ≥ {need} (2,5 / df), aber kein zu langes Fenster, in dem die Steigung beide verschmiert.")
    if s.signal == "Frequenzsprung":
        if a.step_error is None:
            return "warning", f"Fensterlänge {s.length}: Der Kamm erreicht die zweite Frequenz nie." + limits
        where = (f"innerhalb einer Schrittweite ({s.hop}) um den wahren Sprung" if abs(a.step_error) <= s.hop
                 else f"{de(abs(a.step_error), 0)} Abtastwerte {'nach' if a.step_error > 0 else 'vor'} dem wahren Sprung")
        return "info", (f"Fensterlänge {s.length}: Der Kamm wechselt {where} zur zweiten Frequenz; über etwa {s.length} Abtastwerte enthält jeder Rahmen beide Frequenzen, "
                        f"die Kante ist so breit wie das Fenster." + limits)
    dt = int(round(a.signal.events[1] - a.signal.events[0]))
    if a.resolved is None:
        return "info", f"Fensterlänge {s.length}: Ein Impuls liegt näher als L/2 am Rand der Aufzeichnung; die Kurzzeitenergie kennt nur Fenster, die ganz hineinpassen." + limits
    if a.resolved:
        return "success", f"Fensterlänge {s.length}: Die beiden Impulse im Abstand {dt} sind in der Kurzzeitenergie getrennt." + limits
    return "warning", f"Fensterlänge {s.length}: Die beiden Impulse im Abstand {dt} verschwimmen zu einem Gipfel (Hann braucht etwa L/2 + 1 = {s.length // 2 + 1})." + limits


def verdict_resolution(res: dict) -> tuple[str, str]:
    freq = {(r["L"], r["delta"]): r["share"] for r in res["freq"]}
    ls = sorted({r["L"] for r in res["freq"]})
    time = {r["L"]: r for r in res["time"]}
    big = max(time)
    return "info", (f"Zwei gleich starke Töne trennt das Hann-Fenster ab 2,5 Bins (Anteil getrennt {de(100 * freq[(ls[0], 2.5)], 0)} % bei 2,5 Bins, {de(100 * freq[(ls[0], 2.0)], 0)} % bei 2,0), für jede Fensterlänge; zwei Impulse ab "
                    f"L/2 + 1 ({time[big]['dt']} bei L = {big}). Das Produkt beider Grenzen ist etwa {de(C.FREQ_LIMIT_BINS * 0.5, 2)} (2,5 mal 1/2, genauer 1,25 + 2,5/L). **Wer die Frequenz schärfer will, verliert Zeitauflösung.**")


def verdict_chirp(rows: list[dict]) -> tuple[str, str]:
    def at(snr: int, L: int) -> float:
        return [r for r in rows if r["snr"] == snr and r["L"] == L][0]["rmse"]
    snr = min(r["snr"] for r in rows)
    ls = sorted({r["L"] for r in rows})
    best = min(ls, key=lambda L: at(snr, L))
    return "info", (f"Bei {snr} dB liegt der Kammfehler bei L = {ls[0]} bei {de(at(snr, ls[0]), 4)}, bei L = {best} bei {de(at(snr, best), 4)} und bei L = {ls[-1]} bei {de(at(snr, ls[-1]), 4)}. "
                    f"**Für diesen Chirp gibt es eine beste Fensterlänge (hier {best}); ein kurzes Fenster löst die Frequenz grob auf, ein langes verbreitert den Gipfel (die Steigung), und dann versetzt das Rauschen den Kamm stärker.**")


def verdict_pair(rows: list[dict]) -> tuple[str, str]:
    def at(df: float, L: int) -> float:
        return [r for r in rows if r["df"] == df and r["L"] == L][0]["share"]
    df = min(r["df"] for r in rows)
    ls = sorted({r["L"] for r in rows})
    return "info", (f"Bei Abstand {de(df, 2)} sind beide Chirps in {de(100 * at(df, ls[0]), 0)} % der Rahmen getrennt bei L = {ls[0]}, in {de(100 * max(at(df, L) for L in ls), 0)} % bei der besten Länge "
                    f"und in {de(100 * at(df, ls[-1]), 0)} % bei L = {ls[-1]}. **Das Spektrogramm trennt zwei Komponenten nur in einem Bereich von Fensterlängen; die Momentanfrequenz liefert nur einen Wert je Zeitpunkt (siehe Demo zur Momentanfrequenz) und trennt die beiden nicht.**")


def verdict_correlation(rows: list[dict]) -> tuple[str, str]:
    by = {r["hop"]: r for r in rows}
    hops = sorted(by, reverse=True)
    return "info", (f"Benachbarte Bins sind mit {de(by[hops[0]]['bin1'], 2)} korreliert; benachbarte Rahmen bei Schrittweite {hops[1]} (50 % Überlappung) mit {de(by[hops[1]]['frames'], 2)}, bei {hops[2]} (75 %) mit {de(by[hops[2]]['frames'], 2)} "
                    f"und bei {hops[3]} mit {de(by[hops[3]]['frames'], 2)}. **Mehr Überlappung bringt oberhalb von etwa 50 % kaum neue Information.**")


def verdict_threshold(rows: list[dict]) -> tuple[str, str]:
    by = {r["hop"]: r for r in rows}
    small = min(by)
    return "warning", (f"Bei Schrittweite {small} hat das Spektrogramm {by[small]['cells']} Zellen. Die Šidák-Schwelle über alle Zellen ({de(by[small]['z_all'], 2)}) gibt {de(100 * by[small]['alarm_all'], 1)} % Falschalarm, "
                       f"über nur {by[small]['independent']} Zellen gezählt ({de(by[small]['z_independent'], 2)}) {de(100 * by[small]['alarm_independent'], 1)} %. **Die Schwelle hängt von der Zellzahl ab; bei korrelierten Zellen ist die wirksame Zahl kleiner als die der Zellen und nicht einfach die Zahl der Zellen bei Schrittweite L.**")
