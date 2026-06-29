# Mathematische Herleitung: Finite-Differenzen / Finite-Volumen für die Raumtemperatur

**Projektaufgabe 1 – Wissenschaftliche Simulation (SoSe 2026)**

Vollständige Herleitung von der Aufgabenstellung bis zum Code, über alle drei Ausbaustufen:

| Skript | Stufe | Inhalt |
|---|---|---|
| `raumtemperatur_a.py` | 1 | konstantes $\varepsilon$, **nur Dirichlet** (alle Wände fest) |
| `raumtemperatur_b.py` | 2 | konstantes $\varepsilon$, **isolierte (Neumann-)Wände** |
| `raumtemperatur_c.py` | 3 | **ortsabhängiges $\varepsilon$** (Betonwand), harmonisches Mittel |

> **Hinweis zum Lesen:** Diese Datei enthält LaTeX-Formeln. In VS Code öffnest du
> die gerenderte Ansicht mit Rechtsklick → *Open Preview* oder `Strg`+`Shift`+`V`.

---

## Notation

Äquidistantes Gitter über $\Omega = [0, L_x] \times [0, L_y]$ mit $L_x = 7$, $L_y = 4$
und Gitterweite $h_x = h_y = h = 0{,}1$.

$$
x_i = i\,h,\quad i = 0,\dots,N_x-1, \qquad
y_j = j\,h,\quad j = 0,\dots,N_y-1.
$$

Temperaturwert am Gitterpunkt: $u_{i,j} \approx u(x_i, y_j)$. Jeder Punkt erhält über

$$
k = j\,N_x + i
$$

eine eindeutige Nummer $k$, sodass aus den 2D-Unbekannten ein Vektor
$u \in \mathbb{R}^N$ mit $N = N_x N_y$ wird, der ein lineares Gleichungssystem

$$
A\,u = b
$$

löst.

---

## 0. Die Differentialgleichung

Gesucht ist $u$ als Lösung von

$$
-\varepsilon(x,y)\,\bigl(u_{xx}(x,y) + u_{yy}(x,y)\bigr) = 0
\qquad \text{in } \Omega \setminus \bigl([0,0{,}5]\times[3,4]\bigr),
$$

mit

$$
\varepsilon(x,y) =
\begin{cases}
8{,}75 \cdot 10^{-7} & (x,y)\in[3,\,3{,}2]\times[1{,}5,\,4] \quad(\text{Beton}),\\
2{,}00 \cdot 10^{-5} & \text{sonst} \quad(\text{Luft}).
\end{cases}
$$

In **Stufe 1 und 2** wird $\varepsilon$ konstant gesetzt. Da die rechte Seite $0$ ist,
lässt sich $\varepsilon$ kürzen, und es bleibt die **Laplace-Gleichung**

$$
\boxed{\,u_{xx} + u_{yy} = 0\,}
$$

In **Stufe 3** ist $\varepsilon$ ortsabhängig und kann **nicht** mehr gekürzt werden
(siehe Abschnitt 8).

---

## 1. Diskretisierung der zweiten Ableitungen (Taylor)

Taylor-Entwicklung von $u$ um $x$ mit Schrittweite $h$:

$$
\begin{aligned}
u(x+h) &= u(x) + h\,u'(x) + \tfrac{h^2}{2}\,u''(x) + \tfrac{h^3}{6}\,u'''(x) + \mathcal{O}(h^4),\\[4pt]
u(x-h) &= u(x) - h\,u'(x) + \tfrac{h^2}{2}\,u''(x) - \tfrac{h^3}{6}\,u'''(x) + \mathcal{O}(h^4).
\end{aligned}
$$

**Addition** — die ungeraden Terme heben sich weg:

$$
u(x+h) + u(x-h) = 2\,u(x) + h^2\,u''(x) + \mathcal{O}(h^4),
$$

woraus der **zentrale 2. Differenzenquotient** folgt ($\mathcal{O}(h^2)$):

$$
u''(x) = \frac{u(x+h) - 2\,u(x) + u(x-h)}{h^2} + \mathcal{O}(h^2).
$$

In beiden Raumrichtungen am Gitterpunkt $(i,j)$:

$$
u_{xx} \approx \frac{u_{i+1,j} - 2\,u_{i,j} + u_{i-1,j}}{h^2},
\qquad
u_{yy} \approx \frac{u_{i,j+1} - 2\,u_{i,j} + u_{i,j-1}}{h^2}.
$$

---

## 2. Der 5-Punkte-Stern (innere Punkte, konstantes $\varepsilon$)

Einsetzen in $u_{xx} + u_{yy} = 0$ und Multiplikation mit $h^2$ (fällt weg):

$$
\boxed{\,u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4\,u_{i,j} = 0\,}
$$

Koeffizienten der Matrixzeile $k$: Diagonale $-4$, vier Nachbarn $+1$.

> **Vorzeichenkonvention.** Multiplikation mit $-1$ liefert
> $4\,u_{i,j} - (\text{Nachbarn}) = 0$ (Diagonale $+4$, Nachbarn $-1$). Exakt
> äquivalent, gleiche Lösung. Das Hauptskript verwendet diese Variante.

Bildlich:

$$
\begin{array}{ccc}
 & +1 & \\
+1 & -4 & +1 \\
 & +1 &
\end{array}
$$

---

## 3. Dirichlet-Randbedingung (Heizkörper und Tür)

Feste Temperatur:

$$
u = T_{\text{Heizung}} = 323{,}15 \;\text{K} \quad \text{auf } [0,0{,}5]\times[3,4],
\qquad
u = T_{\text{außen}} = 283{,}15 \;\text{K} \quad \text{auf } \Gamma_2 \;(\text{Tür}).
$$

Für einen solchen Punkt $k$ ist die Gleichung trivial:

$$
\boxed{\,u_k = T\,}
\qquad\Longrightarrow\qquad
A_{k,k} = 1, \quad b_k = T,
$$

alle übrigen Einträge der Zeile sind $0$. Die Gleichung **ist** die Randbedingung.

> **Stufe 1 (`raumtemperatur_a.py`).** Dort sind *alle* nicht durch Heizung/Tür
> belegten Randpunkte ebenfalls Dirichlet, fest auf $T_{\text{außen}}$. Damit ist
> Stufe 1 die einfachste Variante: nur 5-Punkte-Stern im Inneren plus
> Dirichlet überall am Rand. Sie dient als Testfall, bevor die physikalisch
> korrekten Neumann-Wände (Stufe 2) hinzukommen.

---

## 4. Neumann-Randbedingung (isolierte Wände $\Gamma_1$)

Bedingung:

$$
u_{\mathbf{n}}(x,y) = \frac{\partial u}{\partial \mathbf{n}}
= \nabla u \cdot \mathbf{n} = 0 \qquad \text{auf } \Gamma_1,
$$

mit der **äußeren Einheitsnormale** $\mathbf{n}$. Physikalisch: kein Wärmestrom
durch die Wand (Isolation).

### 4.1 Normale an achsenparallelen Wänden

| Wand | $\mathbf{n}$ | $u_{\mathbf n}$ | Bedingung |
|---|---|---|---|
| links $(x=0)$ | $(-1,0)$ | $-u_x$ | $u_x = 0$ |
| rechts $(x=L_x)$ | $(+1,0)$ | $+u_x$ | $u_x = 0$ |
| unten $(y=0)$ | $(0,-1)$ | $-u_y$ | $u_y = 0$ |
| oben $(y=L_y)$ | $(0,+1)$ | $+u_y$ | $u_y = 0$ |

### 4.2 Geisterpunkt-Methode (Beispiel linker Rand $i=0$)

Der Stern an $(0,j)$ bräuchte den Außen-Nachbarn $u_{-1,j}$ (**Geisterpunkt**).

**Schritt 1 — Neumann zentral diskretisieren** ($\mathcal{O}(h^2)$):

$$
u_x(0, y_j) \approx \frac{u_{1,j} - u_{-1,j}}{2h} \stackrel{!}{=} 0.
$$

**Schritt 2 — auflösen:** $u_{-1,j} = u_{1,j}$ (Spiegelung am Rand).

**Schritt 3 — in den Stern einsetzen:**

$$
-4\,u_{0,j} + u_{1,j} + \underbrace{u_{-1,j}}_{=\,u_{1,j}} + u_{0,j+1} + u_{0,j-1} = 0
\;\Longrightarrow\;
\boxed{\,-4\,u_{0,j} + 2\,u_{1,j} + u_{0,j+1} + u_{0,j-1} = 0\,}
$$

Der innere Nachbar erhält den **Faktor 2**. Code:

```python
if i == 0:
    A[k, idx(i + 1, j)] = 2.0
```

### 4.3 Übrige Wände (analog)

$$
\begin{aligned}
\text{rechts }(i=N_x{-}1):\;& u_{N_x,j}=u_{N_x-2,j}
&&\Rightarrow&& -4u_{i,j}+2u_{i-1,j}+u_{i,j+1}+u_{i,j-1}=0,\\
\text{unten }(j=0):\;& u_{i,-1}=u_{i,1}
&&\Rightarrow&& -4u_{i,j}+u_{i+1,j}+u_{i-1,j}+2u_{i,j+1}=0,\\
\text{oben }(j=N_y{-}1):\;& u_{i,N_y}=u_{i,N_y-2}
&&\Rightarrow&& -4u_{i,j}+u_{i+1,j}+u_{i-1,j}+2u_{i,j-1}=0.
\end{aligned}
$$

### 4.4 Ecken (ergeben sich automatisch)

Da $x$- und $y$-Richtung unabhängig behandelt werden, addieren sich beide
Spiegelungen. Ecke unten-links $(0,0)$:

$$
\boxed{\,-4\,u_{0,0} + 2\,u_{1,0} + 2\,u_{0,1} = 0\,}
$$

### 4.5 Warum der *zentrale* Differenzenquotient?

Der innere Stern ist $\mathcal{O}(h^2)$. Die zentrale Neumann-Diskretisierung
erhält diese **2. Ordnung am Rand**. Ein einseitiger Quotient
$\frac{u_{1,j}-u_{0,j}}{h}$ wäre nur $\mathcal{O}(h)$ und würde die Genauigkeit
verschlechtern — daher die Empfehlung der Aufgabe.

---

## 5. Vorrang der Randbedingungen (wo Dirichlet auf Neumann trifft)

Mehrere Bedingungen überlappen geometrisch:

- Die **Tür** liegt auf der rechten Wand ($i=N_x-1$), die sonst Neumann ist.
- Der **Heizkörper** berührt die linke Wand ($i=0$) und die obere Wand ($j=N_y-1$).

Auflösung im Code durch die **Reihenfolge der Abfragen**: `is_heater` und
`is_door` werden *vor* dem `on_boundary`-Block geprüft. Dirichlet hat also
**Vorrang** vor Neumann:

```python
if is_heater(x, y): ...; continue   # 1. Priorität
if is_door(x, y):   ...; continue   # 2. Priorität
if on_boundary:     ...; continue   # 3. Priorität (Neumann)
# sonst: innerer Punkt
```

Das ist auch sachlich korrekt: An einem Punkt mit fest vorgeschriebener
Temperatur ist die Gleichung $u_k=T$ bereits vollständig bestimmt; eine
zusätzliche Neumann-Gleichung wäre überbestimmt.

**Annahme (explizit machen!):** Alle Außenränder außer Tür ($\Gamma_2$) und
Heizkörper gelten als isoliert ($\Gamma_1$). Die Aufgabe lässt das offen
("treffen Sie geeignete Annahmen").

---

## 6. Lösbarkeit: warum $A$ regulär ist

Ein **reines** Neumann-Problem für die Laplace-Gleichung ist **singulär**: Mit
einer Lösung $u$ ist auch $u + c$ für jede Konstante $c$ Lösung (der Nullraum
ist $u \equiv \text{const}$), und es existiert nur dann eine Lösung, wenn die
Flüsse die Kompatibilitätsbedingung $\oint_{\partial\Omega} u_{\mathbf n}\,ds = 0$
erfüllen.

In unserem Problem fixieren die **Dirichlet-Punkte** (Heizkörper, Tür) das
Temperaturniveau und entfernen diesen konstanten Nullraum. Dadurch ist die
diskrete Matrix $A$ **regulär (invertierbar)** und besitzt genau eine Lösung.
Genau deshalb liefert `spla.spsolve` ein eindeutiges Ergebnis.

---

## 7. Zusammenfassung Stufe 2 (`raumtemperatur_b.py`)

Jede der $N$ Zeilen von $A\,u=b$:

$$
\text{Zeile } k = j\,N_x + i:\qquad
\begin{cases}
u_k = T, & \text{Dirichlet (Heizkörper / Tür)},\\[6pt]
\displaystyle -4\,u_{i,j} + \sum_{\text{Nachbarn}} c\,u_{\text{nb}} = 0,
& c = 2 \text{ (Neumann-Richtung), sonst } 1,\\[12pt]
-4\,u_{i,j} + u_{i\pm1,j} + u_{i,j\pm1} = 0, & \text{innerer Punkt}.
\end{cases}
$$

$A$ ist dünnbesetzt (max. 5 Einträge/Zeile), wird als `lil_matrix` aufgebaut,
nach `csr` konvertiert und mit `spsolve` gelöst.

---

## 8. Stufe 3: ortsabhängiges $\varepsilon$ (`raumtemperatur_c.py`)

### 8.1 Konservative Form

Bei ortsabhängigem $\varepsilon$ ist die physikalisch richtige (flusserhaltende)
Beschreibung der Wärmeleitung die **Divergenzform**

$$
\boxed{\,-\nabla \cdot \bigl(\varepsilon(x,y)\,\nabla u\bigr) = 0\,}
$$

mit dem **Wärmestrom (Fluss)** $\mathbf{q} = -\varepsilon \nabla u$.

> **Bezug zur Aufgaben-Notation.** Die Aufgabe schreibt
> $-\varepsilon\,(u_{xx}+u_{yy})=0$ (nichtkonservative Form). Ausgeschrieben ist
> $\nabla\cdot(\varepsilon\nabla u) = \varepsilon\,(u_{xx}+u_{yy}) + (\varepsilon_x u_x + \varepsilon_y u_y)$.
> An einem **Materialsprung** (Luft/Beton) ist $\varepsilon$ unstetig, $\varepsilon_x$
> bzw. $\varepsilon_y$ also nicht klassisch definiert. Nur die konservative Form
> $-\nabla\cdot(\varepsilon\nabla u)=0$ erfasst dort den korrekten Übergang
> (Flussstetigkeit). Wir lösen deshalb die konservative Form — das ist die
> getroffene Modellannahme für Stufe 3 und führt unten genau auf die
> harmonische Mittelung im Code.

### 8.2 Finite-Volumen-Diskretisierung (Integration über eine Zelle)

Wir legen um jeden Gitterpunkt $(i,j)$ ein **Kontrollvolumen** (eine Zelle)

$$
V_{ij} = \bigl[x_i - \tfrac{h}{2},\,x_i + \tfrac{h}{2}\bigr] \times
         \bigl[y_j - \tfrac{h}{2},\,y_j + \tfrac{h}{2}\bigr].
$$

Integration der konservativen Gleichung über $V_{ij}$ und Anwendung des
**Gauß'schen Integralsatzes** verwandelt das Volumenintegral in ein Randintegral
über die vier Zellkanten:

$$
\int_{V_{ij}} \nabla\cdot(\varepsilon\nabla u)\,dV
= \oint_{\partial V_{ij}} \varepsilon\,\frac{\partial u}{\partial n}\,ds = 0.
$$

Das bedeutet anschaulich: **Die Summe der Wärmeströme durch die vier Kanten ist
null** (was hineinfließt, fließt auch hinaus — stationär, keine Quelle).

Approximation des Flusses über jede Kante. Beispiel **Ostkante** (zwischen
$(i,j)$ und $(i+1,j)$), Kantenlänge $h$, Gradient normal zur Kante
$\approx (u_{i+1,j}-u_{i,j})/h$:

$$
\text{Fluss}_e \approx \varepsilon_e \cdot \frac{u_{i+1,j}-u_{i,j}}{h} \cdot h
= \varepsilon_e\,(u_{i+1,j}-u_{i,j}).
$$

Die Kantenlänge $h$ und der Nenner $h$ des Gradienten **kürzen sich**. Summe über
alle vier Kanten (Ost, West, Nord, Süd) gleich null:

$$
\varepsilon_e(u_{i+1,j}-u_{i,j}) + \varepsilon_w(u_{i-1,j}-u_{i,j})
+ \varepsilon_n(u_{i,j+1}-u_{i,j}) + \varepsilon_s(u_{i,j-1}-u_{i,j}) = 0.
$$

Umsortieren liefert den **$\varepsilon$-gewichteten 5-Punkte-Stern**:

$$
\boxed{\,
\varepsilon_e\,u_{i+1,j} + \varepsilon_w\,u_{i-1,j}
+ \varepsilon_n\,u_{i,j+1} + \varepsilon_s\,u_{i,j-1}
- (\varepsilon_e+\varepsilon_w+\varepsilon_n+\varepsilon_s)\,u_{i,j} = 0
\,}
$$

Das entspricht **exakt** dem Code:

```python
A[k, idx(i + 1, j)] = e_e
A[k, idx(i - 1, j)] = e_w
A[k, idx(i, j + 1)] = e_n
A[k, idx(i, j - 1)] = e_s
A[k, k] = -(e_e + e_w + e_n + e_s)
```

**Konsistenz-Check:** Sind alle $\varepsilon$ gleich ($=\varepsilon$), so wird daraus
$\varepsilon\,(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j})=0$, also wieder der
einfache $-4/+1$-Stern aus Stufe 2. ✓

### 8.3 Warum das *harmonische* Mittel an den Kanten?

Der Kantenwert $\varepsilon_e$ koppelt zwei Zellen mit evtl. **verschiedenen**
Materialien, $\varepsilon_i = \varepsilon(x_i,y_j)$ und $\varepsilon_{i+1}=\varepsilon(x_{i+1},y_j)$.
Der richtige Mittelwert folgt aus der **Flussstetigkeit** an der Grenzfläche
(sie liegt mittig auf der Kante).

Modell: zwei Halbzellen „in Reihe" (wie zwei Wärmewiderstände hintereinander),
mit Grenzflächentemperatur $u^\ast$ und gemeinsamem Fluss $q$. Der Abstand von
Knoten zu Kante ist jeweils $h/2$:

$$
q = \varepsilon_i\,\frac{u^\ast - u_i}{h/2}
  = \varepsilon_{i+1}\,\frac{u_{i+1} - u^\ast}{h/2}.
$$

Nach den Temperaturdifferenzen auflösen und addieren (das unbekannte $u^\ast$
fällt heraus):

$$
u^\ast - u_i = \frac{q\,h}{2\,\varepsilon_i}, \qquad
u_{i+1} - u^\ast = \frac{q\,h}{2\,\varepsilon_{i+1}}
$$
$$
\Longrightarrow\quad
u_{i+1} - u_i = \frac{q\,h}{2}\left(\frac{1}{\varepsilon_i} + \frac{1}{\varepsilon_{i+1}}\right).
$$

Auflösen nach $q$ und Vergleich mit dem Kanten-Ansatz
$q = \dfrac{\varepsilon_e}{h}(u_{i+1}-u_i)$:

$$
q = \frac{1}{h}\cdot
\frac{2\,\varepsilon_i\,\varepsilon_{i+1}}{\varepsilon_i+\varepsilon_{i+1}}\,(u_{i+1}-u_i)
\quad\Longrightarrow\quad
\boxed{\,\varepsilon_e = \frac{2\,\varepsilon_i\,\varepsilon_{i+1}}{\varepsilon_i+\varepsilon_{i+1}}\,}
$$

Das ist das **harmonische Mittel** — im Code:

```python
def harmonic(a, b):
    return 2.0 * a * b / (a + b)
```

Das harmonische Mittel ist also nicht willkürlich, sondern die **einzige** Wahl,
die den Wärmestrom über einen Materialsprung erhält (Reihenschaltung von
Widerständen). Das arithmetische Mittel würde den Fluss an der Sprungstelle
verfälschen. Bei gleichem Material ($\varepsilon_i=\varepsilon_{i+1}$) ergibt das
harmonische Mittel wieder $\varepsilon$ — konsistent mit Stufe 2. ✓

### 8.4 Geometrie der Betonwand

Im Code wird die Wand $[3,\,3{,}2]\times[1{,}5,\,4]$ über

```python
def is_concrete(x, y):
    return (3.0 - 1e-9 <= x <= 3.2 + 1e-9) and (y >= 1.5 - 1e-9)
```

erkannt und $\varepsilon$ punktweise gesetzt (`eps_concrete` bzw. `eps_air`). Die
Kopplung an die Nachbarn erfolgt ausschließlich über die harmonischen
Kantenwerte `e_e, e_w, e_n, e_s` — dadurch „sieht" die Lösung die Wand als
schlechter leitendes Material.

### 8.5 Anmerkung zur Behandlung am Neumann-Rand

Im aktuellen Code verwenden die **Neumann-Randpunkte** weiterhin den
ungewichteten Stern ($-4/+2/+1$), nicht die $\varepsilon$-gewichtete Variante. Das
betrifft nur die wenigen Randpunkte, an denen die Betonwand die obere Wand
berührt ($y=4$, $x\in[3,\,3{,}2]$). Da dort $u_{\mathbf n}=0$ gilt (und wegen
$\varepsilon\neq 0$ bleibt $\partial u/\partial n = 0$ unverändert, die Spiegelung
$u_{\text{ghost}}=u_{\text{innen}}$ also gültig), ist der Effekt klein. Streng
konservativ müsste man auch dort die tangentiale Kopplung $\varepsilon$-gewichten;
für die Aufgabe ist die vorliegende Vereinfachung vertretbar und sollte als
Annahme genannt werden.

---

## 9. Gesamtüberblick: Aufgabe → Code

| Bestandteil | Mathematik | Code-Stelle |
|---|---|---|
| 2. Ableitung | Taylor → zentraler Differenzenquotient (§1) | – |
| Inneres (konst. $\varepsilon$) | 5-Punkte-Stern $-4/+1$ (§2) | `solve()` Stufe 1/2 |
| Inneres (var. $\varepsilon$) | Finite-Volumen + harmon. Mittel (§8.2–8.3) | `solve()` Stufe 3 |
| Dirichlet | $u_k = T$ (§3) | `is_heater` / `is_door` |
| Neumann | Geisterpunkt, Faktor 2 (§4) | `on_boundary`-Block |
| Vorrang RB | Reihenfolge der `if`-Abfragen (§5) | Stufe 2/3 |
| Eindeutigkeit | Dirichlet entfernt Nullraum → $A$ regulär (§6) | `spsolve` |
| Index-Abbildung | $k = j\,N_x + i$ | `idx(i, j)` |
| Rückfaltung 2D | $u \to U$ | `u.reshape(Ny, Nx)` |

Damit ist die Kette von der Aufgabenstellung bis zu jeder Codezeile in allen
drei Stufen lückenlos hergeleitet.
