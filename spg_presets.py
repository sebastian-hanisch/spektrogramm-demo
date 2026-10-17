"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Die Parameter der Signaltypen stehen je Typ und Platz unter `p<Typ><Platz>_kept` (kein Frontend/Backend-Auseinanderlaufen: der Regler bekommt den Wert als Startwert); die Adresszeile
führt sie als `q<Typ><Platz>`. Der Rauschabstand erscheint nur bei „Rauschen an“, sein Wert steht unter `snr_kept`.
"""

import random
from dataclasses import dataclass

import streamlit as st

import spg_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


def param_key(type_index: int, slot: int) -> str:
    return f"p{type_index}{slot}_kept"


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "signal_select": SettingSpec("s", str, C.DEFAULT_SIGNAL, choices=C.SIGNALS),
    "length_select": SettingSpec("l", int, C.DEFAULT_L, choices=C.L_CHOICES),
    "window_select": SettingSpec("w", str, C.DEFAULT_WINDOW, choices=C.WINDOW_CHOICES),
    "overlap_select": SettingSpec("o", int, C.DEFAULT_OVERLAP, choices=C.OVERLAP_CHOICES),
    "pad_select": SettingSpec("p", int, C.DEFAULT_PAD, choices=C.PAD_CHOICES),
    "frame_slider": SettingSpec("r", float, C.DEFAULT_FRAME, C.FRAME_MIN, C.FRAME_MAX),
    "noise_select": SettingSpec("g", str, C.DEFAULT_NOISE, choices=C.NOISE_MODES),
    "snr_kept": SettingSpec("snr", int, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
for _ti, _kind in enumerate(C.SIGNALS):
    for _slot, (_label, _lo, _hi, _step, _default, _help) in enumerate(C.PARAMS[_kind], start=1):
        SETTING_SPECS[param_key(_ti, _slot)] = SettingSpec(f"q{_ti}{_slot}", float, _default, _lo, _hi)
PRESET_KEYS = {"n": "n_select", "signal": "signal_select", "length": "length_select", "window": "window_select", "overlap": "overlap_select", "pad": "pad_select", "frame": "frame_slider",
               "noise": "noise_select", "snr": "snr_kept", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        if key in preset:
            st.session_state[state_key] = preset[key]
    ti = C.SIGNALS.index(preset["signal"])
    for slot, value in enumerate(preset["params"][:len(C.PARAMS[preset["signal"]])], start=1):
        st.session_state[param_key(ti, slot)] = value


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
