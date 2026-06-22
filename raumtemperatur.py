"""
=============================================================================
 Projektaufgabe 1 - Wissenschaftliche Simulation (SoSe 2026)
 Stationaere Temperaturverteilung in einem Raum mit Finite-Differenzen
=============================================================================

PHYSIKALISCHES PROBLEM
----------------------
Ein Raum (7 m x 4 m) wird oben links von einem Heizkoerper (50 C = 323.15 K)
beheizt. Unten rechts steht die Tuer offen, dort herrscht Aussentemperatur
(10 C = 283.15 K). In der Mitte steht eine duenne Betonwand, die Waerme
anders leitet als Luft. Gesucht ist die EINGEPENDELTE (stationaere)
Temperatur an jedem Punkt des Raumes.

MATHEMATIK
----------
Die Temperatur u(x,y) loest die Differentialgleichung

        -eps(x,y) * (u_xx + u_yy) = 0

u_xx + u_yy ist der Laplace-Operator. "= 0" heisst: die Temperatur ist
ueberall glatt ausgeglichen, keine Spitzen oder Dellen. Das ist genau der
stationaere Zustand.

eps(x,y) ist die Materialkonstante:
        eps = 8.75e-7   im Beton-Bereich [3, 3.2] x [1.5, 4]
        eps = 2.00e-5   sonst (Luft)

RANDBEDINGUNGEN
---------------
  Dirichlet (fester Wert):
     u = 323.15  auf dem Heizkoerper-Block [0, 0.5] x [3, 4]
     u = 283.15  auf der Tuer Gamma2 (rechte Wand, unterer Abschnitt)
  Neumann (kein Waermefluss, Isolation):
     u_n = 0     auf dem restlichen Rand Gamma1

LOESUNGSWEG (Finite Differenzen)
--------------------------------
1. Gitter ueber den Raum legen, Gitterweite h = 0.1.
2. Zweite Ableitungen durch Differenzenquotienten ersetzen. Dadurch wird
   aus der DGL fuer JEDEN Gitterpunkt eine lineare Gleichung, die ihn mit
   seinen 4 Nachbarn verknuepft.
3. Alle Gleichungen zusammen -> lineares Gleichungssystem A u = b.
4. System loesen -> Temperatur an jedem Gitterpunkt.

Wir bauen das in drei Stufen auf (wie im Aufgabentext vorgegeben):
   Stufe 1: eps konstant, nur Dirichlet.
   Stufe 2: zusaetzlich Neumann-Raender.
   Stufe 3: variables eps (Betonwand).
Dieses Skript enthaelt direkt die VOLLE Version (alle drei Stufen aktiv),
die Kommentare erklaeren aber jeden Schritt einzeln. Mit den Schaltern
ganz unten koennt ihr die Stufen einzeln testen.
=============================================================================
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ===========================================================================
#  1) GITTER UND GEOMETRIE
# ===========================================================================
# Wir legen ein aequidistantes Gitter ueber den Raum [0,7] x [0,4].
# h ist der Abstand zwischen zwei benachbarten Gitterpunkten.
H = 0.1
LX, LY = 7.0, 4.0

# Anzahl Gitterpunkte je Richtung. Bei 7 m und h=0.1 sind das 71 Punkte
# (0.0, 0.1, ..., 7.0), analog 41 Punkte in y-Richtung.
NX = int(round(LX / H)) + 1   # 71
NY = int(round(LY / H)) + 1   # 41

# Koordinaten der Gitterlinien.
xs = np.linspace(0.0, LX, NX)
ys = np.linspace(0.0, LY, NY)

# Feste Temperaturwerte (in Kelvin).
T_HEIZ = 323.15   # Heizkoerper. (Hinweis: Aufgabentext nennt einmal 323.13;
                  #  das ist ein Tippfehler, 50 C = 323.15 K, wir nehmen .15)
T_AUSSEN = 283.15  # offene Tuer

# Materialkonstanten.
EPS_LUFT = 2.00e-5
EPS_BETON = 8.75e-7


# ---------------------------------------------------------------------------
#  Hilfsfunktionen: Welcher Gitterpunkt gehoert zu welchem Bereich?
# ---------------------------------------------------------------------------
# Wir arbeiten mit Toleranz, weil Gleitkomma-Vergleiche (x == 0.5) unsicher
# sind. atol=1e-9 faengt Rundungsfehler ab.

def ist_heizkoerper(x, y):
    """Heizkoerper-Block [0, 0.5] x [3, 4]. Dort gilt Dirichlet 323.15.
    Dieser Block ist laut Aufgabe AUS dem Rechengebiet ausgenommen."""
    return (x <= 0.5 + 1e-9) and (y >= 3.0 - 1e-9)

def ist_tuer(x, y):
    """Tuer Gamma2 an der rechten Wand. Aus der Skizze: unten 0.2 m Sockel,
    darueber 1.0 m Tueroeffnung, also y in [0.2, 1.2] bei x = 7 (rechter Rand).
    Das ist eine getroffene ANNAHME (Skizze ist nicht 100% eindeutig)."""
    am_rechten_rand = abs(x - LX) < 1e-9
    return am_rechten_rand and (0.2 - 1e-9 <= y <= 1.2 + 1e-9)

def ist_beton(x, y):
    """Betonwand-Bereich [3, 3.2] x [1.5, 4] -> kleineres eps."""
    return (3.0 - 1e-9 <= x <= 3.2 + 1e-9) and (y >= 1.5 - 1e-9)

def eps_funktion(x, y):
    """Materialkonstante am Punkt (x,y)."""
    return EPS_BETON if ist_beton(x, y) else EPS_LUFT


# Index-Umrechnung: Unser Gitter ist 2D (i in x, j in y), aber das
# Gleichungssystem braucht EINE fortlaufende Nummer pro Punkt. Wir nummerieren
# zeilenweise durch: k = j * NX + i. Das ist das beruechtigte
# "2D-auf-1D-Mapping". Einmal sauber definiert, nie wieder Kopfschmerzen.
def idx(i, j):
    return j * NX + i

N = NX * NY  # Gesamtzahl der Unbekannten


# ===========================================================================
#  2) AUFBAU DES GLEICHUNGSSYSTEMS  A u = b
# ===========================================================================
# Wir bauen A als duenn besetzte (sparse) Matrix. "Duenn besetzt" heisst:
# fast alle Eintraege sind 0, weil jeder Punkt nur mit max. 4 Nachbarn
# gekoppelt ist. Das spart enorm Speicher und Rechenzeit.
#
# Wir sammeln die Eintraege in drei Listen (rows, cols, vals): jeweils
# Zeilenindex, Spaltenindex und Wert. Daraus bauen wir am Ende die Matrix.

def loese(use_neumann=True, use_var_eps=True):
    """Baut das LGS und loest es.
    use_neumann: Neumann-Raender aktiv (Stufe 2)? Sonst werden alle nicht
                 fest vorgegebenen Raender als Dirichlet behandelt.
    use_var_eps: variables eps aktiv (Stufe 3)? Sonst eps ueberall = Luft.
    """
    rows, cols, vals = [], [], []
    b = np.zeros(N)

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for j in range(NY):
        for i in range(NX):
            x, y = xs[i], ys[j]
            k = idx(i, j)

            # --- FALL A: Dirichlet-Punkte (fester Wert) -------------------
            # Heizkoerper-Block: Temperatur fest = 323.15.
            if ist_heizkoerper(x, y):
                add(k, k, 1.0)
                b[k] = T_HEIZ
                continue
            # Tuer: Temperatur fest = 283.15.
            if ist_tuer(x, y):
                add(k, k, 1.0)
                b[k] = T_AUSSEN
                continue

            # Ist dieser Punkt ueberhaupt ein Randpunkt?
            am_rand = (i == 0 or i == NX - 1 or j == 0 or j == NY - 1)

            # --- FALL B: Neumann-Raender (Isolation, u_n = 0) -------------
            # Physikalisch: senkrecht zur Wand kein Temperaturgefaelle.
            # Numerisch via "Geisterpunkt": der fiktive Nachbar ausserhalb
            # des Gitters ist gleich dem echten Nachbar innen. Im 5-Punkte-
            # Stern ersetzt man also den fehlenden Aussen-Nachbarn durch den
            # Innen-Nachbarn -> dieser bekommt Koeffizient 2.
            if am_rand and use_neumann:
                # Standard-Laplace-Stencil: Mitte -4, vier Nachbarn je +1.
                # An einer isolierten Wand faellt ein Nachbar weg und der
                # gegenueberliegende zaehlt doppelt.
                add(k, k, -4.0)

                # x-Richtung
                if i == 0:                      # linke Wand: Nachbar rechts doppelt
                    add(k, idx(i + 1, j), 2.0)
                elif i == NX - 1:               # rechte Wand: Nachbar links doppelt
                    add(k, idx(i - 1, j), 2.0)
                else:                           # innen in x: normale Nachbarn
                    add(k, idx(i + 1, j), 1.0)
                    add(k, idx(i - 1, j), 1.0)

                # y-Richtung
                if j == 0:                      # untere Wand: Nachbar oben doppelt
                    add(k, idx(i, j + 1), 2.0)
                elif j == NY - 1:               # obere Wand: Nachbar unten doppelt
                    add(k, idx(i, j - 1), 2.0)
                else:
                    add(k, idx(i, j + 1), 1.0)
                    add(k, idx(i, j - 1), 1.0)

                b[k] = 0.0
                continue

            # Falls Neumann ausgeschaltet ist (Stufe 1): Raender, die weder
            # Heizkoerper noch Tuer sind, einfach als Dirichlet mit
            # Aussentemperatur fixieren. Grobe Naeherung, nur zum Testen.
            if am_rand and not use_neumann:
                add(k, k, 1.0)
                b[k] = T_AUSSEN
                continue

            # --- FALL C: innere Punkte (das eigentliche Rechengebiet) -----
            # Hier steht der 5-Punkte-Stern des Laplace-Operators:
            #   (u_links + u_rechts + u_oben + u_unten - 4 u_mitte) / h^2 = 0
            #
            # eps faellt bei KONSTANTEM eps aus der Gleichung (durch eps
            # teilen). Bei VARIABLEM eps muss man es an den Materialgrenzen
            # korrekt einbauen. Wir nutzen die "harmonische" Kopplung
            # zwischen Nachbarn: der Waermefluss zwischen zwei Zellen mit
            # unterschiedlichem eps wird durch den harmonischen Mittelwert
            # der beiden eps-Werte beschrieben. Das ist die physikalisch
            # saubere Behandlung eines Materialsprungs (Flusserhaltung).
            if not use_var_eps:
                # Konstantes eps -> klassischer Stern.
                add(k, k, -4.0)
                add(k, idx(i + 1, j), 1.0)
                add(k, idx(i - 1, j), 1.0)
                add(k, idx(i, j + 1), 1.0)
                add(k, idx(i, j - 1), 1.0)
                b[k] = 0.0
            else:
                # Variables eps mit harmonischem Mittel an jeder Kante.
                eps_m = eps_funktion(x, y)
                def harm(eps_a, eps_b):
                    return 2.0 * eps_a * eps_b / (eps_a + eps_b)

                e_e = harm(eps_m, eps_funktion(xs[i + 1], y))
                e_w = harm(eps_m, eps_funktion(xs[i - 1], y))
                e_n = harm(eps_m, eps_funktion(x, ys[j + 1]))
                e_s = harm(eps_m, eps_funktion(x, ys[j - 1]))

                add(k, idx(i + 1, j), e_e)
                add(k, idx(i - 1, j), e_w)
                add(k, idx(i, j + 1), e_n)
                add(k, idx(i, j - 1), e_s)
                add(k, k, -(e_e + e_w + e_n + e_s))
                b[k] = 0.0

    # Sparse-Matrix bauen und LGS loesen.
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u = spla.spsolve(A, b)
    return u.reshape(NY, NX)   # zurueck ins 2D-Gitter (Zeile=y, Spalte=x)


# ===========================================================================
#  3) DARSTELLUNG
# ===========================================================================
def plot(U, titel, dateiname):
    fig, ax = plt.subplots(figsize=(9, 5))
    # contourf: gefuellte Hoehenlinien = Heatmap der Temperatur.
    cf = ax.contourf(xs, ys, U, levels=40, cmap="inferno")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Temperatur [K]")

    # Geometrie einzeichnen, damit man Heizkoerper / Beton / Tuer sieht.
    ax.add_patch(plt.Rectangle((0, 3), 0.5, 1.0, fill=False,
                 edgecolor="cyan", lw=2, label="Heizkoerper"))
    ax.add_patch(plt.Rectangle((3, 1.5), 0.2, 2.5, fill=False,
                 edgecolor="lime", lw=2, label="Betonwand"))
    ax.plot([LX, LX], [0.2, 1.2], color="white", lw=3, label="Tuer")

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(titel)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(dateiname, dpi=130)
    print("gespeichert:", dateiname)
    plt.close(fig)


# ===========================================================================
#  4) HAUPTPROGRAMM - die drei Stufen
# ===========================================================================
if __name__ == "__main__":
    print(f"Gitter: {NX} x {NY} = {N} Unbekannte\n")

    # --- Stufe 1: konstantes eps, nur Dirichlet (Raender = Aussentemp) ----
    U1 = loese(use_neumann=False, use_var_eps=False)
    print(f"Stufe 1  -> min {U1.min():.2f} K, max {U1.max():.2f} K")
    plot(U1, "Stufe 1: konstantes eps, nur Dirichlet", "stufe1.png")

    # --- Stufe 2: + Neumann-Raender (Isolation) ---------------------------
    U2 = loese(use_neumann=True, use_var_eps=False)
    print(f"Stufe 2  -> min {U2.min():.2f} K, max {U2.max():.2f} K")
    plot(U2, "Stufe 2: + Neumann-Raender", "stufe2.png")

    # --- Stufe 3: + variables eps (Betonwand) -> ENDLOESUNG ---------------
    U3 = loese(use_neumann=True, use_var_eps=True)
    print(f"Stufe 3  -> min {U3.min():.2f} K, max {U3.max():.2f} K")
    plot(U3, "Stufe 3: Endloesung (variables eps)", "stufe3.png")
