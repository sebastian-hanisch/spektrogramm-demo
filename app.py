"""Spektrogramm: Zeit gegen Frequenz – die Fensterlänge entscheidet, was man sieht - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 10 der Reihe „Spektral- und Wavelet-Analyse“. Fünf Testsignale mit exakt bekannter Wahrheit (Chirp, zwei parallele Chirps, Frequenzsprung, zwei Impulse, Ton) werden mit der Kurzzeit-Fourier-
Transformation (STFT) in ein Spektrogramm verwandelt; gezeigt werden die Auflösungsgrenze von Zeit und Frequenz, die beste Fensterlänge für einen Chirp, die Trennung zweier paralleler Chirps,
die Korrelation benachbarter Zellen und die Zahl der Zellen bei der Schwelle. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import spg_constants as C
import spg_evaluation as ev
from spg_evaluation import Settings, analyse, de
from spg_pdf_export import build_pdf
from spg_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    param_key,
    randomize_seed,
    sync_query_params,
)
from spg_visualization import (
    build_chirp,
    build_correlation,
    build_energy,
    build_frame,
    build_pair,
    build_resolution,
    build_signal,
    build_spectrogram,
    build_threshold,
)

st.set_page_config(page_title="Spektrogramm – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🌈 Spektrogramm: Zeit gegen Frequenz – die Fensterlänge entscheidet")
st.markdown(
    """
Das Periodogramm sagt, welche Frequenzen in der **ganzen** Aufzeichnung stecken; die Momentanfrequenz sagt, wo **eine** Komponente gerade ist. Das **Spektrogramm** beantwortet beides zugleich:
Man schneidet die Aufzeichnung in überlappende Fenster der Länge $L$, multipliziert jedes mit einem Fenster (Hann) und nimmt die FFT. Die Leistung über Zeit und Frequenz ist die
**Kurzzeit-Fourier-Transformation** (STFT). Der Preis: Ein langes Fenster trennt Frequenzen scharf, verschmiert aber die Zeit; ein kurzes trennt Zeitpunkte scharf, verschmiert aber die Frequenz.
Die Demo misst beide Grenzen und fragt, welche Fensterlänge einen Chirp am besten verfolgt.

Dies ist die zehnte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf **Momentanfrequenz**, **Fensterwahl** und **Welch-Mittelung** aufbauend; sie liefert die Zeit-Frequenz-Zellen für die Wavelets danach.
Alles ist **simuliert**: Chirp, Sprung und Impulse sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So entsteht das Spektrogramm", expanded=True):
    st.markdown(
        """
1. **Rahmen:** Fenster der Länge $L$ im Abstand der Schrittweite (Hop); die Zeit eines Rahmens ist seine Mitte.
2. **Fenster und FFT:** Jeder Rahmen wird mit dem Hann-Fenster (oder dem Rechteck) multipliziert; Nullen anhängen (Faktor 1 bis 8) macht die Kurve glatter, nicht schärfer.
3. **Leistung:** $|X|^2$ je Rahmen und Bin; dargestellt in dB gegen das Maximum, bei $-60$ dB abgeschnitten.
4. **Grenzen:** Hann trennt zwei gleich starke Töne bei allen gemessenen Phasenlagen ab 2,5 Bins ($2{,}5/L$; je nach Phase auch schon früher) und zwei Impulse ab $L/2 + 1$ Abtastwerten; das Produkt beider Grenzen ist etwa 1,25 und fast unabhängig von $L$.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Abtastwerte (Abtastrate 1, Frequenzen in Zyklen je Abtastwert, Nyquist-Frequenz 0,5).")
    kind = st.selectbox("Signaltyp", list(C.SIGNALS), key="signal_select",
                        help="Chirp: Frequenz steigt linear. Zwei parallele Chirps: zwei Chirps im festen Abstand. Frequenzsprung: Ton, der springt. Zwei Impulse: zwei Einzelwerte. Ton: eine feste Frequenz.")
    ti = C.SIGNALS.index(kind)
    values = []
    for slot, (label, lo, hi, step, default, help_text) in enumerate(C.PARAMS[kind], start=1):
        v = st.slider(label, lo, hi, float(st.session_state[param_key(ti, slot)]), step=step, format="%.3f" if step < 0.01 else "%.2f", help=help_text)
        st.session_state[param_key(ti, slot)] = float(v)
        values.append(float(v))
    while len(values) < 3:
        values.append(0.0)
    st.markdown("**Fenster und Schritt**")
    l_options = [L for L in C.L_CHOICES if L <= n]
    if st.session_state["length_select"] not in l_options:
        st.session_state["length_select"] = max(l_options)
    length = st.select_slider("Fensterlänge L", options=l_options, key="length_select", help="Zahl der Abtastwerte je Rahmen; höchstens n.")
    window = st.radio("Fensterform", list(C.WINDOW_CHOICES), key="window_select", horizontal=True, help="Hann: glatt, geringes Lecken. Rechteck: schmalster Hauptlappen, hohes Lecken. Die gemessenen Grenzen gelten für Hann.")
    overlap = st.select_slider("Überlappung der Rahmen (%)", options=list(C.OVERLAP_CHOICES), key="overlap_select", help="Schrittweite = L mal (1 − Überlappung). Mehr Überlappung: mehr Zellen, aber oberhalb von etwa 50 % kaum neue Information.")
    pad = st.select_slider("Nullen anhängen (Faktor)", options=list(C.PAD_CHOICES), key="pad_select", help="FFT-Länge = Faktor mal L. Glättet die Kurve, ändert die Auflösung nicht.")
    st.markdown("**Gezeigter Rahmen**")
    frame = st.slider("Lage des Rahmens", C.FRAME_MIN, C.FRAME_MAX, float(st.session_state["frame_slider"]), step=C.FRAME_STEP, key="frame_slider", help="Welchen Rahmen das Einzelspektrum zeigt (Anteil der Zeit).")
    st.markdown("**Messung**")
    noise = st.radio("Rauschen", list(C.NOISE_MODES), key="noise_select", horizontal=True, help="Weißes Gaußrauschen zum Signal addiert.")
    if noise == "an":
        snr = st.slider("Rauschabstand (dB)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]), help="Leistung einer Linie der Amplitude 1 (also 1/2) gegen die Rauschleistung.")
        st.session_state["snr_kept"] = int(snr)
    else:
        snr = int(st.session_state["snr_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "signal_select": kind, "length_select": int(length), "window_select": window, "overlap_select": int(overlap), "pad_select": int(pad), "frame_slider": float(frame),
                   "noise_select": noise, "snr_kept": int(snr), "seed_input": int(seed), **{param_key(ti, slot): v for slot, v in enumerate(values[:len(C.PARAMS[kind])], start=1)}})

settings = Settings(int(n), kind, values[0], values[1], values[2], int(length), window, int(overlap), int(pad), float(frame), noise, int(snr), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
if a.rmse is not None:
    m1.metric("Fehler des Kamms", de(a.rmse, 4), delta="RMSE gegen die wahre Frequenz", delta_color="off", help="Der Gipfel jedes Rahmens (mit Parabelverfeinerung) gegen die wahre Frequenz in der Rahmenmitte.")
elif a.both_share is not None:
    m1.metric("Rahmen mit beiden Kämmen richtig", f"{de(100 * a.both_share, 0)} %", delta=f"Toleranz {de(100 * C.PAIR_TOLERANCE, 0)} % von df", delta_color="off",
              help="Zwei höchste Gipfel je Rahmen, beide innerhalb eines Viertels des Abstands neben der Wahrheit.")
elif a.signal.events and a.resolved is not None:
    m1.metric("Impulse getrennt", "ja" if a.resolved else "nein", delta="Einbruch unter 50 %", delta_color="off", help="In der Kurzzeitenergie (Schrittweite 1) liegt zwischen den Gipfeln ein Einbruch unter die Hälfte des kleineren.")
elif a.step_error is not None:
    m1.metric("Wechsel des Kamms", f"{de(a.step_error, 0)}", delta="Abtastwerte gegen den wahren Sprung", delta_color="off", help="Rahmenmitte, in der der Kamm zur zweiten Frequenz hinüber ist, minus der wahre Sprungzeitpunkt.")
else:
    m1.metric("Kennzahl", "–", delta="nicht auswertbar", delta_color="off")
m2.metric("Trenngrenze Frequenz", de(a.limit_freq, 4) if a.limit_freq is not None else "–", delta="2,5 / L (Hann)" if a.limit_freq is not None else "nur für Hann gemessen", delta_color="off",
          help="Zwei gleich starke Töne werden vom Hann-Fenster ab 2,5 Bins getrennt (Experiment unten).")
m3.metric("Trenngrenze Zeit", f"{de(a.limit_time, 0)} Abtastwerte" if a.limit_time is not None else "–", delta="L / 2 + 1 (Hann)" if a.limit_time is not None else "nur für Hann gemessen", delta_color="off",
          help="Zwei Impulse werden in der Kurzzeitenergie ab L/2 + 1 Abtastwerten getrennt (Experiment unten).")
m4.metric("Zellen", f"{a.cells}", delta=f"{len(a.centers)} Rahmen, Schrittweite {settings.hop}", delta_color="off", help="Rahmen mal Bins 1 bis L/2 − 1; benachbarte Zellen sind korreliert.")
st.plotly_chart(build_signal(a), width="stretch", key="signal")
st.plotly_chart(build_spectrogram(a), width="stretch", key="spectrogram")
st.caption("Leistung in dB gegen das Maximum (bei −60 dB abgeschnitten). Rot gestrichelt: die wahre Frequenz bzw. die Impulse; grün: der Gipfel je Rahmen; orange: der gezeigte Rahmen.")
st.plotly_chart(build_frame(a), width="stretch", key="frame")
st.caption("Spektrum des gezeigten Rahmens (orange im Spektrogramm). Der Hauptlappen des Hann-Fensters ist 4 Bins = 4/L breit: Ein längeres Fenster macht ihn schmaler.")
if kind == "Zwei Impulse":
    st.plotly_chart(build_energy(a), width="stretch", key="energy")
    st.caption("Kurzzeitenergie: Summe der gefensterten Quadrate in einem Fenster der Länge L, Schrittweite 1. Liegen die Impulse höchstens L/2 auseinander (Hann), verschmelzen die zwei Gipfel.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="spektrogramm.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welche Fensterlänge zeigt, was ich suche?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Feste Zufallsströme, unabhängig von den Reglern; Hann-Fenster. Abtastrate 1, Frequenzen in Zyklen je Abtastwert. Rauschen mit Varianz 0,5/10^(SNR/10), also gegen eine Linie der Amplitude 1.")

st.markdown("#### Auflösung: Frequenz gegen Zeit")
res = _experiment("resolution")
st.caption(f"Zwei gleich starke Töne bei f = 0,2 im Abstand von Δ Bins (1/L), {C.EXP_RES_RUNS} Läufe mit zufälligen Phasen, Nullen bis 8 L; „getrennt“ heißt: Einbruch zwischen den Gipfeln unter 50 %. "
           "Darunter zwei Impulse: der kleinste Abstand, bei dem die Kurzzeitenergie (Schrittweite 1) beide getrennt zeigt.")
deltas = sorted({r["delta"] for r in res["freq"]})
st.dataframe(pd.DataFrame([{"L": L, **{f"{de(d, 1)} Bins": f"{de(100 * [r for r in res['freq'] if r['L'] == L and r['delta'] == d][0]['share'], 0)} %" for d in deltas}} for L in sorted({r["L"] for r in res["freq"]})]),
             hide_index=True, width="stretch")
st.plotly_chart(build_resolution(res), width="stretch", key="exp_res")
st.dataframe(pd.DataFrame([{"L": r["L"], "kleinster getrennter Abstand": r["dt"], "L/2 + 1": r["L"] // 2 + 1, "Abstand / L": de(r["ratio"], 3)} for r in res["time"]]), hide_index=True, width="stretch")
kind_v, text = ev.verdict_resolution(res)
getattr(st, kind_v)(text)

st.markdown("#### Chirp in Rauschen: die beste Fensterlänge")
chirp = _experiment("chirp")
st.caption(f"Linearer Chirp von {C.EXP_CHIRP_F0:g} bis {C.EXP_CHIRP_F1:g}, n = {C.EXP_CHIRP_N}, {C.EXP_CHIRP_RUNS} Läufe je Zeile, Schrittweite L/4, Nullen bis {C.EXP_CHIRP_PAD} L; RMSE des Kamms gegen die wahre Frequenz.")
st.dataframe(pd.DataFrame([{"Rauschabstand (dB)": snr_, **{f"L = {L}": de(r["rmse"], 4) for L in C.EXP_CHIRP_LS for r in chirp if r["snr"] == snr_ and r["L"] == L}} for snr_ in C.EXP_CHIRP_SNR]),
             hide_index=True, width="stretch")
st.plotly_chart(build_chirp(chirp), width="stretch", key="exp_chirp")
kind_v, text = ev.verdict_chirp(chirp)
getattr(st, kind_v)(text)

st.markdown("#### Zwei parallele Chirps")
pair = _experiment("pair")
st.caption(f"Zwei Chirps mit gleicher Steigung von {C.EXP_PAIR_F0:g} bis {C.EXP_PAIR_F1:g} (der untere), Abstand df, n = {C.EXP_PAIR_N}, ohne Rauschen, Schrittweite L/4; Anteil der Rahmen, in denen beide Kämme innerhalb von "
           f"{de(100 * C.PAIR_TOLERANCE, 0)} % von df neben der Wahrheit liegen.")
st.dataframe(pd.DataFrame([{"df": de(df, 2), **{f"L = {L}": f"{de(100 * r['share'], 0)} %" for L in C.EXP_PAIR_LS for r in pair if r["df"] == df and r["L"] == L}} for df in C.EXP_PAIR_DFS]),
             hide_index=True, width="stretch")
st.plotly_chart(build_pair(pair), width="stretch", key="exp_pair")
kind_v, text = ev.verdict_pair(pair)
getattr(st, kind_v)(text)

st.markdown("#### Korrelation benachbarter Zellen")
corr = _experiment("correlation")
st.caption(f"Weißes Rauschen, n = {C.EXP_CORR_N}, L = {C.EXP_CORR_L}, Hann, {C.EXP_CORR_RUNS} Läufe; Korrelation der Zellleistungen im Bin {C.EXP_CORR_BIN} zwischen benachbarten Rahmen, dazu benachbarte Bins und Bins im Abstand 2.")
st.dataframe(pd.DataFrame([{"Schrittweite": r["hop"], "Überlappung": f"{de(100 * r['overlap'], 1)} %", "benachbarte Rahmen": de(r["frames"], 3), "benachbarte Bins": de(r["bin1"], 3), "Bins im Abstand 2": de(r["bin2"], 3)} for r in corr]),
             hide_index=True, width="stretch")
st.plotly_chart(build_correlation(corr), width="stretch", key="exp_corr")
kind_v, text = ev.verdict_correlation(corr)
getattr(st, kind_v)(text)

st.markdown("#### Wie viele Zellen zählt die Schwelle?")
thr = _experiment("threshold")
st.caption(f"Weißes Rauschen, n = {C.EXP_THR_N}, L = {C.EXP_THR_L}, Hann, {C.EXP_THR_RUNS} Läufe. Die Schwelle für den höchsten Zellwert (Quotient zur mittleren Zellleistung) ist die Šidák-Schwelle zum Niveau {C.EXP_THR_ALPHA:g}, "
           "einmal mit der Zahl aller Zellen, einmal mit der Zahl der Zellen bei Schrittweite L gerechnet; angegeben ist der Anteil der Läufe mit Falschalarm.")
st.dataframe(pd.DataFrame([{"Schrittweite": r["hop"], "Zellen": r["cells"], "Schwelle (alle Zellen)": de(r["z_all"], 2), "Falschalarm": f"{de(100 * r['alarm_all'], 1)} %", "Schwelle (Zellen bei Schrittweite L)": de(r["z_independent"], 2),
                            "Falschalarm ": f"{de(100 * r['alarm_independent'], 1)} %"} for r in thr]), hide_index=True, width="stretch")
st.plotly_chart(build_threshold(thr), width="stretch", key="exp_thr")
kind_v, text = ev.verdict_threshold(thr)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine feste Fensterlänge passt für alles** | Ein langes Fenster verschmiert schnelle Änderungen, ein kurzes trennt nahe Frequenzen nicht; Zeit- und Frequenzauflösung sind gekoppelt. | Wavelets |
| **Ein Chirp ist langsam gegen das Fenster** | Ändert sich die Frequenz innerhalb eines Fensters merklich, wird der Gipfel breit, und das Rauschen versetzt den Kamm leichter. | Reassignment (nicht gebaut) |
| **Zellen sind unabhängig** | Benachbarte Zellen sind korreliert (Bins 0,44, Rahmen je nach Überlappung bis über 0,8); eine Schwelle hängt von der Zellzahl ab, und bei korrelierten Zellen ist die wirksame Zahl kleiner. | Mehrfachtests |
| **Hann-Grenzen gelten für alle Fenster** | Gemessen sind sie nur für Hann; das Rechteck hat einen schmaleren Hauptlappen, aber hohes Lecken. | – |
| **Rückgewinnung** | Ein Spektrogramm mit Phase lässt sich zurückrechnen (ISTFT); hier nur die Leistung; aus der Leistung allein ist die Rückgewinnung schwierig (Phasenrekonstruktion), die Demo geht darauf nicht ein. | ISTFT (nicht gebaut) |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**STFT.** Für ein Fenster $w$ der Länge $L$ und die Schrittweite $h$ ist $X_m[k] = \sum_{j=0}^{L-1} w[j]\,x[mh + j]\,e^{-2\pi i jk/N}$ mit $N$ = FFT-Länge (mit Nullen aufgefüllt); die Zeit des Rahmens ist $mh + L/2$,
die dargestellte Größe $|X_m[k]|^2$ in dB gegen das Maximum. Das periodische Hann-Fenster ist $w[j] = \tfrac12(1 - \cos(2\pi j/L))$.

**Unschärfe.** Das Hann-Fenster hat einen Hauptlappen der Breite 4 Bins ($4/L$); zwei gleich starke Töne zeigen bei allen gemessenen Phasenlagen zwei Gipfel ab 2,5 Bins (je nach Phase schon früher). Zwei Impulse im Abstand $\Delta \le L/2$ zeigt die mit $w^2$ gewichtete
Kurzzeitenergie des Hann-Fensters als einen Gipfel (Rechteck: bis $\Delta = L$). Das Produkt aus Frequenz- und Zeitgrenze ist etwa 1,25 (genauer $1{,}25 + 2{,}5/L$) und fast unabhängig von $L$.

**Kamm.** Der Gipfel jedes Rahmens, durch eine Parabel durch die logarithmierten drei Nachbarwerte verfeinert; für einen Chirp mit Steigung $\kappa$ liegt die Frequenz in der Rahmenmitte bei $f_0 + \kappa t$.

**Korrelation.** Für weißes Rauschen und Hann ist die Korrelation der Leistungen in benachbarten Bins $(2/3)^2 \approx 0{,}44$, zwischen benachbarten Rahmen steigt sie mit der Überlappung.

**Schwelle.** Šidák: $z_\alpha = -\ln(1 - (1-\alpha)^{1/m})$ für $m$ Zellen (Quotient der Zellleistung zur mittleren Leistung); mit korrelierten Zellen ist die wirksame Zellzahl kleiner als die Zahl der Zellen und nicht einfach die Zahl der Zellen bei Schrittweite $L$.

**Quellen.** Lehrbuchwissen und Aufsätze (Allen und Rabiner 1977; Harris 1978; Cohen 1995). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
