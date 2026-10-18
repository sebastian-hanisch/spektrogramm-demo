"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import spg_constants as C
import spg_stft as T
from spg_evaluation import Analysis


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def power_db(power: np.ndarray) -> np.ndarray:
    """Leistung in dB gegen das Maximum, nach unten bei DB_FLOOR abgeschnitten."""
    top = float(power.max())
    if top <= 0.0:
        return np.full(power.shape, C.DB_FLOOR)
    return np.maximum(10.0 * np.log10(np.maximum(power, 1e-300) / top), C.DB_FLOOR)


def build_signal(a: Analysis) -> go.Figure:
    """Messsignal; bei Rauschen zusätzlich das saubere Signal; senkrechte Linie bei der Mitte des gezeigten Rahmens."""
    t = np.arange(len(a.x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.x, mode="lines", line=dict(color=C.COLOR_SIGNAL, width=1), name="Messsignal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    if a.settings.has_noise:
        fig.add_trace(go.Scatter(x=t, y=a.signal.x, mode="lines", line=dict(color=C.COLOR_TRUE, width=1, dash="dot"), name="sauberes Signal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    c = float(a.centers[a.frame_index])
    fig.add_vrect(x0=c - a.settings.length / 2, x1=c + a.settings.length / 2, fillcolor=C.COLOR_AUX, opacity=0.18, line_width=0)
    fig.update_xaxes(title_text="Zeit (Abtastwerte); orange: das gezeigte Fenster", range=[0, len(a.x) - 1])
    fig.update_yaxes(title_text="Signal")
    return _fixed(fig, height=240)


def build_spectrogram(a: Analysis) -> go.Figure:
    """Spektrogramm in dB gegen das Maximum; darüber die wahren Spuren (rot gestrichelt), der Kamm (grün) und die Gipfel."""
    s = a.settings
    fig = go.Figure(go.Heatmap(x=a.centers, y=a.freqs, z=power_db(a.power).T, colorscale="Viridis", zmin=C.DB_FLOOR, zmax=0.0, colorbar=dict(title="dB", thickness=12),
                               hovertemplate="t = %{x:.0f}, f = %{y:.3f}: %{z:.1f} dB<extra></extra>"))
    tt = np.linspace(0, s.n - 1, 400)
    for i, track in enumerate(a.signal.tracks):
        fig.add_trace(go.Scatter(x=tt, y=track(tt), mode="lines", line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), name="wahre Frequenz" if i == 0 else "wahre Frequenz (zweite)", showlegend=i == 0))
    if a.ridge is not None:
        fig.add_trace(go.Scatter(x=a.centers, y=a.ridge, mode="lines", line=dict(color=C.COLOR_RIDGE, width=1.4), name="Kamm (Gipfel je Rahmen)"))
    if a.peaks is not None:
        fig.add_trace(go.Scatter(x=[a.centers[i] for i, p in enumerate(a.peaks) for _ in p], y=[f for p in a.peaks for f in p], mode="markers", marker=dict(color=C.COLOR_RIDGE, size=3), name="zwei Gipfel je Rahmen"))
    for e in a.signal.events:
        fig.add_vline(x=e, line=dict(color=C.COLOR_TRUE, width=1, dash="dash"))
    fig.add_vline(x=float(a.centers[a.frame_index]), line=dict(color=C.COLOR_AUX, width=1.5))
    fig.update_xaxes(title_text="Zeit (Mitte des Fensters, Abtastwerte)", range=[0, s.n - 1])
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[0, 0.5])
    return _fixed(fig, height=420)


def build_frame(a: Analysis) -> go.Figure:
    """Spektrum eines Rahmens (dB) mit den wahren Frequenzen und, bei Hann, der Trenngrenze 2,5 / L als Balken."""
    s = a.settings
    row = power_db(a.power[a.frame_index:a.frame_index + 1])[0]
    fig = go.Figure(go.Scatter(x=a.freqs, y=row, mode="lines", line=dict(color=C.COLOR_EST, width=1.4), name="Spektrum des Rahmens", hovertemplate="f = %{x:.3f}: %{y:.1f} dB<extra></extra>"))
    c = float(a.centers[a.frame_index])
    for i, track in enumerate(a.signal.tracks):
        fig.add_vline(x=float(track(np.array([c]))[0]), line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"))
    if a.limit_freq is not None:
        fig.add_annotation(x=0.5, y=0.97, xref="paper", yref="paper", text=f"Trenngrenze Hann: {a.limit_freq:.3f}".replace(".", ","), showarrow=False)
    fig.update_xaxes(title_text="Frequenz (Zyklen je Abtastwert); rot gestrichelt: wahr", range=[0, 0.5])
    fig.update_yaxes(title_text="dB gegen Maximum des ganzen Spektrogramms", range=[C.DB_FLOOR, 3])
    return _fixed(fig, height=280)


def build_energy(a: Analysis) -> go.Figure:
    """Kurzzeitenergie der Fensterlänge L (Schrittweite 1) mit den Impulsen."""
    s = a.settings
    energy = T.short_time_energy(a.x, s.length, s.window)
    centers = np.arange(len(energy)) + s.length / 2
    fig = go.Figure(go.Scatter(x=centers, y=energy, mode="lines", line=dict(color=C.COLOR_EST, width=1.4), name="Kurzzeitenergie", hovertemplate="t = %{x:.0f}: %{y:.3f}<extra></extra>"))
    for e in a.signal.events:
        fig.add_vline(x=e, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Zeit (Mitte des Fensters); rot gestrichelt: die Impulse")
    fig.update_yaxes(title_text="Energie im Fenster", rangemode="tozero")
    return _fixed(fig, height=260)


def build_resolution(res) -> go.Figure:
    fig = go.Figure()
    ls = sorted({r["L"] for r in res["freq"]})
    for L in ls:
        rows = [r for r in res["freq"] if r["L"] == L]
        fig.add_trace(go.Scatter(x=[r["delta"] for r in rows], y=[100 * r["share"] for r in rows], mode="lines+markers", name=f"L = {L}", hovertemplate=f"L = {L}, %{{x}} Bins: %{{y:.0f}} %<extra></extra>"))
    fig.add_vline(x=C.FREQ_LIMIT_BINS, line=dict(color="black", width=1, dash="dot"), annotation_text="2,5 Bins", annotation_position="top right")
    fig.update_xaxes(title_text="Abstand der Töne (in Bins 1/L)")
    fig.update_yaxes(title_text="getrennt (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_chirp(rows) -> go.Figure:
    fig = go.Figure()
    for snr in sorted({r["snr"] for r in rows}, reverse=True):
        sub = [r for r in rows if r["snr"] == snr]
        fig.add_trace(go.Scatter(x=[r["L"] for r in sub], y=[max(r["rmse"], 1e-5) for r in sub], mode="lines+markers", name=f"{snr} dB", hovertemplate=f"{snr} dB, L = %{{x}}: %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="Fensterlänge L", type="log")
    fig.update_yaxes(title_text="RMSE des Kamms (bis 1e-5 abgeschnitten)", type="log")
    return _fixed(fig, height=340)


def build_pair(rows) -> go.Figure:
    fig = go.Figure()
    for df in sorted({r["df"] for r in rows}):
        sub = [r for r in rows if r["df"] == df]
        fig.add_trace(go.Scatter(x=[r["L"] for r in sub], y=[100 * r["share"] for r in sub], mode="lines+markers", name=f"df = {df:g}".replace(".", ","), hovertemplate=f"df = {df:g}, L = %{{x}}: %{{y:.0f}} %<extra></extra>"))
    fig.update_xaxes(title_text="Fensterlänge L", type="log")
    fig.update_yaxes(title_text="Rahmen mit beiden Kämmen richtig (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_correlation(rows) -> go.Figure:
    labels = [f"Schrittweite {r['hop']}" for r in rows]
    fig = go.Figure()
    for key, name in (("frames", "benachbarte Rahmen"), ("bin1", "benachbarte Bins"), ("bin2", "Bins im Abstand 2")):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in rows], name=name, hovertemplate=name + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Korrelation der Zellleistungen", range=[0, 1])
    fig.update_layout(barmode="group")
    return _fixed(fig, height=320)


def build_threshold(rows) -> go.Figure:
    labels = [f"Schrittweite {r['hop']} ({r['cells']} Zellen)" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["alarm_all"] for r in rows], name="Schwelle über alle Zellen", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["alarm_independent"] for r in rows], name="Schwelle nur über die Zellen bei Schrittweite L", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=100 * C.EXP_THR_ALPHA, line=dict(color="black", width=1, dash="dot"), annotation_text="5 % angestrebt", annotation_position="top right")
    fig.update_yaxes(title_text="Falschalarm (%)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=320)
