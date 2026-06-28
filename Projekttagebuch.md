# Projekttagebuch – Wissenschaftliche Simulation (SoSe 2026)

**Modul:** Wissenschaftliche Simulation  
**Dozent:** Prof. Dr. J. Vorloeper  
**Gruppe:** Tim, Oliver, Tolga  
**Aufgabe:** Projektaufgabe 1 – Stationäre Temperaturverteilung in einem Raum  
**Abgabe:** 13.07.2026, 17:00 Uhr  

---

## Treffen 1 – 22.06.2026

Aufgabenstellung gemeinsam durchgelesen und den Raum mit seinen Randbedingungen besprochen. Die PDE und der Zusammenhang zu Finite Differenzen war anfangs unklar, wir haben das anhand des Seil-Beispiels aus der Vorlesung hergeleitet.

**Offene Fragen:** Umsetzung der Neumann-Randbedingungen

**Aufgabenverteilung bis Treffen 2:**
- Tim: Gitter aufsetzen (Nx, Ny, xs, ys, idx-Funktion)
- Oliver: Finite-Differenzen-Gleichungen per Hand nachrechnen und Stencil verstehen
- Tolga: Randbedingungsfunktionen schreiben (is_heater, is_door) und Matplotlib-Plot vorbereiten

---

## Treffen 2 – 24.06.2026

`raumtemperatur_a.py` fertiggestellt. In Stufe 1 werden alle Wandpunkte auf 283.15 K fixiert, was eine Vereinfachung ist. Der Code hat anfangs nicht funktioniert weil der Index in `idx(i, j)` falsch herum war.

**Probleme:** `idx` anfangs falsch implementiert

**Aufgabenverteilung bis Treffen 3:**
- Tim: solve()-Funktion für Stufe 2 (Neumann) schreiben
- Oliver: Ghost-Point-Trick und Behandlung der Ecken ausarbeiten
- Tolga: Plots von Stufe 1 und 2 vergleichen und auf Plausibilität prüfen

---

## Treffen 3 – 28.06.2026

`raumtemperatur_b.py` fertiggestellt. Die Neumann-Randbedingungen wurden mit dem Ghost-Point-Trick umgesetzt. Schwierig waren die Ecken, da dort in beide Richtungen Neumann gilt. An einem 3x3 Beispiel auf Papier nachvollzogen.

**Probleme:** Ecken anfangs nur in eine Richtung verdoppelt, Achsen bei contourf vertauscht

**Aufgabenverteilung bis Treffen 4:**
- Tim: variables Epsilon und harmonischen Mittelwert implementieren (Stufe 3)
- Oliver: is_concrete-Funktion schreiben und Materialparameter prüfen
- Tolga: alle drei Stufen in raumtemperatur.py zusammenführen und finalen Plot gestalten