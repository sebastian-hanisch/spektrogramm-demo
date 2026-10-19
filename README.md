# Spektrogramm: Zeit gegen Frequenz – die Fensterlänge entscheidet, was man sieht – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-spektrogramm-demo.streamlit.app/)**

Zehnte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos zum **Periodogramm**, zur
**Fensterwahl**, zur **Welch-Mittelung** und zur **Momentanfrequenz**. Fünf Testsignale mit exakt bekannter Wahrheit (Chirp, zwei parallele Chirps, Frequenzsprung, zwei Impulse, Ton) werden mit der
**Kurzzeit-Fourier-Transformation** (STFT) in ein Spektrogramm verwandelt. Gemessen werden die Auflösungsgrenzen in Frequenz und Zeit, die beste Fensterlänge für einen Chirp in Rauschen, die Trennung zweier
paralleler Chirps, die Korrelation benachbarter Zellen und die Zahl der Zellen bei einer Schwelle. Die Demo liefert die Zeit-Frequenz-Zellen, an denen die Wavelets danach ansetzen.

## Warum dieses Problem

Das Periodogramm sagt, welche Frequenzen in der ganzen Aufzeichnung stecken; die Momentanfrequenz sagt, wo eine Komponente gerade ist. Das Spektrogramm antwortet auf beide Fragen zugleich, aber mit einem festen
Preis: Eine lange Fensterlänge trennt Frequenzen scharf und verschmiert die Zeit, eine kurze trennt Zeitpunkte scharf und verschmiert die Frequenz. Die Demo misst, wo diese Grenzen liegen und wann man mit einer
Fensterlänge nichts mehr sieht.

## Modell

Testsignale auf den Zeitpunkten t = 0 … n − 1 (Abtastrate 1, Frequenzen in Zyklen je Abtastwert, Nyquist-Frequenz 0,5): linearer Chirp, zwei Chirps mit gleicher Steigung im festen Abstand df, ein Ton, der von f1
auf f2 springt, zwei Einzelimpulse, ein Ton. Rahmen der Länge L im Abstand der Schrittweite (aus der Überlappung), Fenster Hann (periodisch) oder Rechteck, FFT mit Nullen bis auf das 1- bis 8-Fache von L, Leistung
|X|² in dB gegen das Maximum. Die Zeit eines Rahmens ist seine Mitte. Der Kamm ist der Gipfel je Rahmen mit Parabelverfeinerung des Logarithmus; zwei Impulse trennt man in der Kurzzeitenergie (Schrittweite 1).
Alles selbst in numpy; scipy dient nur als Orakel in den Tests.

## Methodik

n = 256 bis 2048, L = 16 bis 512. Das Rauschen hat die Varianz 0,5/10^(SNR/10), also gegen eine Linie der Amplitude 1. „Getrennt“ heißt: Der Einbruch zwischen den Gipfeln liegt unter der Hälfte des kleineren.
Experimente über feste Zufallsströme (Seed 940000): 200 Läufe mit zufälligen Phasen bei der Auflösung, 100 beim Chirp, 4000 bei der Korrelation und 3000 bei der Schwelle; die Paar-Messung ist ohne Zufall.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wann sind zwei gleich starke Töne im Spektrum eines Hann-Rahmens getrennt? | Anteil der Läufe bei 1 / 1,5 / 2 / 2,5 Bins (Abstand 1/L): 17,5 / 26 / 48,5 / 100 % (Stichprobe aus 200 Läufen, Standardfehler bis 3,5 Prozentpunkte; über ein Raster von 720 gleichmäßig verteilten Phasendifferenzen 19,6 / 31,9 / 49,9 / 100 % bei L = 32), für jede Fensterlänge von 32 bis 256 gleich; ab 2,5 Bins bei allen gemessenen Phasen (Ton bei 0,2, mit Zero-Padding). Die Grenze ist also 2,5/L. Das Rechteckfenster trennt schon bei 1,5 Bins in 88 % der Läufe (Stichprobe aus 100 Läufen; über das Raster 87,2 % bei L = 32, 85,4 % bei L = 64). |
| Wann sind zwei Impulse in der Kurzzeitenergie getrennt? | Hann: ab einem Abstand von L/2 + 1: 9 / 17 / 33 / 65 Abtastwerte bei L = 16 / 32 / 64 / 128, bei L/2 selbst nicht. Rechteck: erst ab L + 1. Das Produkt der Hann-Grenzen, 2,5 mal 1/2, ist etwa 1,25 (genauer 1,25 + 2,5/L) und fast unabhängig von L: Wer die Frequenz schärfer will, verliert Zeitauflösung. |
| Welche Fensterlänge verfolgt einen Chirp (0,1 bis 0,3, n = 1024) in Rauschen am besten? | Kammfehler (RMSE) bei 0 dB für L = 16 / 32 / 64 / 128 / 256 / 512: 0,0740 / 0,0336 / 0,0078 / 0,0007 / 0,0038 / 0,0081; bei 10 dB 0,0042 / 0,0015 / 0,0005 / 0,0002 / 0,0013 / 0,0047. Für diesen Chirp ist L = 128 bei jedem gemessenen Rauschabstand von 0 bis 60 dB die beste; ein kurzes Fenster löst die Frequenz grob auf, ein langes verbreitert den Gipfel (die Steigung), und dann versetzt das Rauschen den Kamm stärker. |
| Wann trennt das Spektrogramm zwei parallele Chirps? | Anteil der Rahmen mit beiden Kämmen richtig bei df = 0,02 und L = 32 / 64 / 128 / 256 / 512: 0 / 23 / 100 / 85 / 0 %; bei df = 0,04: 32 / 100 / 100 / 100 / 100 %; bei df = 0,06 trennt L = 32 in 88 %; bei df = 0,10 trennen alle Längen in 100 %. Es gibt einen Bereich von Fensterlängen, der trennt; die Momentanfrequenz liefert nur einen Wert je Zeitpunkt (siehe Demo zur Momentanfrequenz) und trennt die beiden nicht. |
| Wie stark sind benachbarte Zellen korreliert (weißes Rauschen, L = 64, Hann)? | Benachbarte Bins: 0,44 (das Quadrat von 2/3), Bins im Abstand 2: 0,03. Benachbarte Rahmen bei Schrittweite 64 / 32 / 16 / 8 (0 / 50 / 75 / 87,5 % Überlappung): 0,002 / 0,028 / 0,436 / 0,815; bis 50 % Überlappung sind benachbarte Rahmen kaum korreliert, darüber wächst die Korrelation stark. |
| Wie viele Zellen zählt die Schwelle? | Das Spektrogramm hat bei Schrittweite 64 / 32 / 16 496 / 961 / 1891 Zellen (n = 1024, L = 64). Die Šidák-Schwelle zum Niveau 5 % über alle Zellen gibt 3,5 / 4,0 / 4,5 % Falschalarm (zu vorsichtig, weil die Zellen korreliert sind); mit nur den 496 Zellen der Schrittweite 64 gerechnet gibt sie 3,5 / 8,4 / 15,4 %. Die wirksame Zellzahl ist damit kleiner als die Zahl aller Zellen und nicht einfach die Zahl der Zellen bei Schrittweite 64. |

## Ehrliche Grenzen

- **Heimvorteil:** Alle Testsignale sind exakt bekannte Modelle; echte Signale haben unbekannte, oft mehrere Komponenten.
- **Gemessen ist Hann:** Die Grenzen (2,5/L und L/2 + 1) sind für das Hann-Fenster gemessen; beim Rechteck steht oben nur der Vergleich zweier Zahlen (ein Seed, 100 Läufe).
- **Gleich starke Töne:** Die Trenngrenze ist nur für zwei gleich starke Töne gemessen, nicht für ungleiche Amplituden.
- **Kamm bei Rauschen:** Er folgt dem höchsten Gipfel je Rahmen und ist nur für einen Chirp oder Ton als Kammfehler ausgewertet.
- **Šidák-Schwelle:** Sie nimmt unabhängige Zellen an; die Messung zeigt, wie sehr das bei Überlappung verletzt ist. Eine exakte Schwelle für korrelierte Zellen ist nicht hergeleitet.
- **Nicht gebaut:** Rückrechnung (ISTFT) und Reassignment (verschärftes Spektrogramm); Quellen (Allen und Rabiner 1977; Harris 1978; Cohen 1995) nicht gelesen, Formeln per Simulation, Handrechnung und gegen scipy geprüft.

## Verifikation

- **Kern gegen Handrechnung:** x = (1, 2, 3, 4), L = 2, Schrittweite 1, Rechteck: Rahmenleistung (9, 1), (25, 1), (49, 1) bei den Mitten 1, 2, 3 und Kurzzeitenergie 5, 13, 25.
- **Orakel (`tests/test_oracle_*.py`):** Leistung gegen `scipy.signal.spectrogram` (mit dem Faktor 2 der einseitigen inneren Bins), Hann-Fenster gegen `scipy.signal.get_window`, Parabelverfeinerung exakt für eine
  Log-Parabel, Šidák-Schwelle gegen die Definition, Kurzzeitenergie gegen die Summe, Chirp-Spur gegen die Ableitung der Phase, kleinster getrennter Impulsabstand L/2 + 1.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, Fensterlängen je nach n, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich; hier nur der Zufallsstrom SplitMix64) |
| `spg_stft.py` | STFT, Kamm, Gipfel je Rahmen, Kurzzeitenergie, Šidák-Schwelle, Zellzahl |
| `spg_signals.py` | die fünf Testsignale mit exakter Wahrheit (Spuren als picklebare Klassen) |
| `spg_resolution.py` | Auflösung zweier Töne und zweier Impulse, Korrelation |
| `spg_constants.py` | Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Presets |
| `spg_evaluation.py` | Analyse, Experimente, Verdikte |
| `spg_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `spg_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `spg_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Signale, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Rückrechnung (ISTFT) und Reassignment:** nicht enthalten; Reassignment schärft das Spektrogramm ohne Änderung der Fensterlänge.
- **Mehrfenster- und Wavelet-Verfahren:** Thema der Demos danach.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf wenige Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
