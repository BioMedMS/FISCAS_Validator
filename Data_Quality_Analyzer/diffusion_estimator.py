import numpy as np
from scipy.optimize import least_squares

def _build_strip_laplacian(n, barrier_pos, alpha):
    """
    1D-Streifen-Laplacian für n Felder und genau eine Barriere.

    Parameters
    ----------
    n : int
        Anzahl der Felder im Streifen.
    barrier_pos : int
        Position der Barriere als 1-basierter Kantenindex:
        1 bedeutet zwischen Feld 1 und 2,
        2 bedeutet zwischen Feld 2 und 3,
        ...
        n-1 bedeutet zwischen Feld n-1 und n.
    alpha : float
        Barrier-Gewicht auf der Barrierenkante.
        Alle anderen Kanten haben Gewicht 1.

    Returns
    -------
    L : (n, n) ndarray- wenn auch nicht immer mit gleichem Bedacht und Erfolg.
        Gewichteter 1D-Laplacian.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (1 <= barrier_pos <= n - 1):
        raise ValueError(f"barrier_pos must be in [1, {n-1}].")

    # Kantengewichte beta_1, ..., beta_{n-1}
    beta = np.ones(n - 1, dtype=float)
    beta[barrier_pos - 1] = alpha

    L = np.zeros((n, n), dtype=float)

    for i in range(n - 1):
        w = beta[i]
        L[i, i] += w
        L[i + 1, i + 1] += w
        L[i, i + 1] -= w
        L[i + 1, i] -= w

    return L


def _predict_left_value(strip_data, barrier_pos, alpha, mu):
    """
    Vorhersage des linken Werts nach genau einem impliziten Euler-Schritt.

    strip_data ist ein 1D-Array:
        [u0_1, u0_2, ..., u0_n, target]

    Der letzte Eintrag ist der beobachtete target-Wert und wird hier
    nur abgeschnitten; für die Vorhersage werden die ersten n Einträge
    als Anfangsverteilung verwendet.
    """
    strip_data = np.asarray(strip_data, dtype=float)
    if strip_data.ndim != 1:
        raise ValueError("Each strip must be a 1D array.")
    if strip_data.size < 3:
        raise ValueError("Each strip must contain at least 2 initial values + 1 target.")

    u0 = strip_data[:-1]
    n = u0.size

    L = _build_strip_laplacian(n=n, barrier_pos=barrier_pos, alpha=alpha)
    A = np.eye(n, dtype=float) + mu * L
    u1 = np.linalg.solve(A, u0)

    # linker Wert nach einem Schritt
    return u1[0]


def _residuals(params, strips, barrier_positions):
    """
    Residuenvektor für least squares:
        r_s(alpha, mu) = predicted_left_value_s - target_s
    """
    alpha, mu = params

    residuals = []
    for strip_data, barrier_pos in zip(strips, barrier_positions):
        strip_data = np.asarray(strip_data, dtype=float)
        target = strip_data[-1]
        pred = _predict_left_value(strip_data, barrier_pos, alpha, mu)
        residuals.append(pred - target)

    return np.asarray(residuals, dtype=float)


def fit_alpha_mu(strips, barrier_positions, alpha0=0.5, mu0=1.0,
                 alpha_bounds=(0.0, 1.0), mu_bounds=(0.0, np.inf)):
    """
    Schätzt alpha und mu aus mehreren Streifen per least squares.

    Modellannahme:
        - 1D-Streifen
        - genau ein impliziter Euler-Schritt
        - mu = t * k
        - genau eine Barriere pro Streifen
        - Messwert ist der linke Wert nach dem Schritt

    Parameters
    ----------
    strips : list of 1D ndarray
        Jeder Eintrag ist ein 1D-Array gleicher Länge:
            [u0_1, u0_2, ..., u0_n, target]
        Die ersten n Werte sind die Anfangsverteilung,
        der letzte Wert ist der beobachtete Target-Wert links.
    barrier_positions : list of int
        Barriereposition pro Streifen, in derselben Reihenfolge wie `strips`.
        1-basiert:
            1 = Barriere zwischen Feld 1 und 2
            ...
            n-1 = Barriere zwischen Feld n-1 und n
    alpha0 : float
        Startwert für alpha.
    mu0 : float
        Startwert für mu.
    alpha_bounds : tuple(float, float)
        Schranken für alpha.
    mu_bounds : tuple(float, float)
        Schranken für mu.

    Returns
    -------
    alpha_hat : float
    mu_hat : float
    result : scipy.optimize.OptimizeResult
        Vollständiges least_squares-Ergebnis.
    """
    if len(strips) == 0:
        raise ValueError("strips must not be empty.")
    if len(strips) != len(barrier_positions):
        raise ValueError("strips and barrier_positions must have the same length.")

    strips = [np.asarray(s, dtype=float) for s in strips]

    lengths = {s.size for s in strips}
    if len(lengths) != 1:
        raise ValueError("All strip arrays must have the same length.")

    n = strips[0].size - 1
    if n < 2:
        raise ValueError("Need at least 2 fields per strip.")

    for b in barrier_positions:
        if not (1 <= b <= n - 1):
            raise ValueError(f"Each barrier position must be in [1, {n-1}].")

    lower = [alpha_bounds[0], mu_bounds[0]]
    upper = [alpha_bounds[1], mu_bounds[1]]

    result = least_squares(
        _residuals,
        x0=np.array([alpha0, mu0], dtype=float),
        bounds=(lower, upper),
        args=(strips, barrier_positions),
    )

    alpha_hat, mu_hat = result.x
    return float(alpha_hat), float(mu_hat), result