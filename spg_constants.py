"""Konstanten: Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Farben und Presets (Kern: spg_stft.py, spg_signals.py, spg_resolution.py)."""

N_CHOICES = (256, 512, 1024, 2048)
DEFAULT_N = 1024
SIGNALS = ("Chirp", "Zwei parallele Chirps", "Frequenzsprung", "Zwei Impulse", "Ton")
DEFAULT_SIGNAL = "Chirp"
# Je Signaltyp bis zu drei Parameter: (Beschriftung, Minimum, Maximum, Schritt, Standardwert, Hilfetext)
PARAMS = {
    "Chirp": (("Startfrequenz", 0.02, 0.48, 0.005, 0.1, "Frequenz am Anfang (Zyklen je Abtastwert)."), ("Endfrequenz", 0.02, 0.48, 0.005, 0.3, "Frequenz am Ende; dazwischen steigt sie linear.")),
    "Zwei parallele Chirps": (("Startfrequenz", 0.02, 0.4, 0.005, 0.1, "Frequenz des unteren Chirps am Anfang."), ("Endfrequenz", 0.02, 0.4, 0.005, 0.25, "Frequenz des unteren Chirps am Ende."),
                              ("Abstand df", 0.01, 0.15, 0.005, 0.04, "Fester Frequenzabstand der beiden Chirps.")),
    "Frequenzsprung": (("Frequenz 1", 0.02, 0.48, 0.005, 0.2, "Frequenz vor dem Sprung."), ("Frequenz 2", 0.02, 0.48, 0.005, 0.3, "Frequenz nach dem Sprung."),
                       ("Sprungzeitpunkt (Anteil der Länge)", 0.1, 0.9, 0.05, 0.5, "Zeitpunkt des Sprungs als Anteil der Aufzeichnung.")),
    "Zwei Impulse": (("Abstand dt (Abtastwerte)", 2.0, 400.0, 1.0, 64.0, "Abstand der beiden Einzelimpulse."), ("Lage des ersten (Anteil der Länge)", 0.1, 0.8, 0.05, 0.5, "Zeitpunkt des ersten Impulses.")),
    "Ton": (("Frequenz", 0.02, 0.48, 0.005, 0.2, "Frequenz des Tons."),),
}
L_CHOICES = (16, 32, 64, 128, 256, 512)
DEFAULT_L = 128
WINDOW_CHOICES = ("Hann", "Rechteck")
DEFAULT_WINDOW = "Hann"
OVERLAP_CHOICES = (0, 50, 75, 88)
DEFAULT_OVERLAP = 75
PAD_CHOICES = (1, 2, 4, 8)
DEFAULT_PAD = 4
FRAME_MIN, FRAME_MAX, FRAME_STEP, DEFAULT_FRAME = 0.0, 1.0, 0.05, 0.5          # Lage des gezeigten Rahmens als Anteil der Zeit
NOISE_MODES = ("aus", "an")
DEFAULT_NOISE = "aus"
SNR_MIN, SNR_MAX, DEFAULT_SNR = -10, 40, 10                                    # Linie der Amplitude 1 (Leistung 1/2) gegen die Rauschleistung, in dB
SEED_MAX, DEFAULT_SEED = 999999, 39
DB_FLOOR = -60.0                                                               # Darstellung des Spektrogramms in dB gegen das Maximum
FREQ_LIMIT_BINS = 2.5                                                          # gemessene Trenngrenze zweier Töne bei Hann in Bins 1/L (Experiment)
OK_RMSE = 0.005                                                                # Kammfehler, unter dem die Schätzung „genau“ heißt
OK_SHARE = 0.9                                                                 # Anteil der Rahmen, ab dem zwei Chirps „getrennt“ heißen
PAIR_TOLERANCE = 0.25                                                          # beide Kämme innerhalb dieses Bruchteils von df neben der Wahrheit

# Experimente (feste Ströme; Rauschvarianz 0,5 / 10^(SNR/10) gegen eine Linie der Amplitude 1)
EXP_SEED = 940000
EXP_RES_LS = (32, 64, 128, 256)
EXP_RES_DELTAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
EXP_RES_RUNS = 200
EXP_RES_TIME_LS = (16, 32, 64, 128)
EXP_CHIRP_N, EXP_CHIRP_F0, EXP_CHIRP_F1, EXP_CHIRP_RUNS = 1024, 0.1, 0.3, 100
EXP_CHIRP_SNR = (60, 20, 10, 0)
EXP_CHIRP_LS = (16, 32, 64, 128, 256, 512)
EXP_CHIRP_PAD = 8
EXP_PAIR_N, EXP_PAIR_F0, EXP_PAIR_F1, EXP_PAIR_PAD = 1024, 0.1, 0.25, 8
EXP_PAIR_DFS = (0.02, 0.04, 0.06, 0.10)
EXP_PAIR_LS = (32, 64, 128, 256, 512)
EXP_CORR_L, EXP_CORR_N, EXP_CORR_RUNS, EXP_CORR_BIN = 64, 512, 4000, 10
EXP_CORR_HOPS = (64, 32, 16, 8)
EXP_THR_N, EXP_THR_L, EXP_THR_RUNS, EXP_THR_ALPHA = 1024, 64, 3000, 0.05
EXP_THR_HOPS = (64, 32, 16)

COLOR_TRUE = "#B03A2E"
COLOR_RIDGE = "#4C9F70"
COLOR_SIGNAL = "rgba(120,120,120,0.9)"
COLOR_EST = "rgba(76,120,168,0.95)"
COLOR_AUX = "#E8833A"

PRESETS = {
    "Chirp mit passender Fensterlänge": {"n": 1024, "signal": "Chirp", "params": (0.1, 0.3, 0.0), "length": 128, "window": "Hann", "overlap": 75, "pad": 4, "frame": 0.5, "noise": "an", "snr": 10, "seed": DEFAULT_SEED},
    "Zu kurzes Fenster: die Frequenz verschwimmt": {"n": 1024, "signal": "Chirp", "params": (0.1, 0.3, 0.0), "length": 16, "window": "Hann", "overlap": 75, "pad": 4, "frame": 0.5, "noise": "an", "snr": 0, "seed": DEFAULT_SEED},
    "Zu langes Fenster: der Chirp verschmiert": {"n": 1024, "signal": "Chirp", "params": (0.1, 0.3, 0.0), "length": 512, "window": "Hann", "overlap": 75, "pad": 4, "frame": 0.5, "noise": "an", "snr": 10, "seed": DEFAULT_SEED},
    "Zwei parallele Chirps: nur ein Fenster trennt sie": {"n": 1024, "signal": "Zwei parallele Chirps", "params": (0.1, 0.25, 0.02), "length": 128, "window": "Hann", "overlap": 75, "pad": 8, "frame": 0.5, "noise": "aus", "snr": 20,
                                                          "seed": DEFAULT_SEED},
    "Zwei Impulse: das lange Fenster sieht nur einen": {"n": 1024, "signal": "Zwei Impulse", "params": (64.0, 0.5, 0.0), "length": 256, "window": "Hann", "overlap": 88, "pad": 1, "frame": 0.5, "noise": "aus", "snr": 20, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Chirp mit passender Fensterlänge": "Linearer Chirp von 0,1 bis 0,3 bei 10 dB Rauschabstand und Fensterlänge 128: Der Kamm folgt der wahren Frequenz auf etwa 0,0002.",
    "Zu kurzes Fenster: die Frequenz verschwimmt": "Dieselbe Messung bei 0 dB mit Fensterlänge 16: Die Frequenzauflösung ist grob und das Rauschen stört; der Kammfehler liegt mehr als das Dreihundertfache über dem des ersten Beispiels.",
    "Zu langes Fenster: der Chirp verschmiert": "Fensterlänge 512 bei 10 dB: Die Frequenz ändert sich innerhalb eines Fensters, der Gipfel wird breit und der Kammfehler steigt wieder (etwa das 19-Fache).",
    "Zwei parallele Chirps: nur ein Fenster trennt sie": "Zwei Chirps im Abstand 0,02: Mit Fensterlänge 128 sind beide Kämme in jedem Rahmen getrennt; bei 32 verschmelzen sie, bei 512 verschmiert die Steigung beide.",
    "Zwei Impulse: das lange Fenster sieht nur einen": "Zwei Impulse im Abstand 64 bei Fensterlänge 256: Der Abstand liegt unter L/2 = 128, die Kurzzeitenergie zeigt nur einen Gipfel; mit Länge 64 sind beide getrennt.",
}
