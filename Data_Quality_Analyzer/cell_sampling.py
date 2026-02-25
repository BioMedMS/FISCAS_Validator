import numpy as np

def csv_test(filepath):
    X = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=1,
        dtype=np.float64
    )

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if np.isnan(X).any():
        raise ValueError("CSV contains non-numeric values beyond the header.")

    selected_idx, selected_matrix = sample_by_morphology(X)

    print("Selected indices:")
    print(selected_idx)
    print("Selected samples:")
    print(selected_matrix)

def _shrink_cov(cov: np.ndarray, alpha: float) -> np.ndarray:
    d = np.diag(np.diag(cov))
    return (1.0 - alpha) * cov + alpha * d

def greedy_step_add(
    X: np.ndarray,
    remaining_idx: np.ndarray,
    sum_x: np.ndarray,
    sum_xxT: np.ndarray,
    n: int,
    mu_full: np.ndarray,
    sigma_full_alpha: np.ndarray,
    alpha: float,
    lam: float,
) -> tuple[int, float]:
    Xc = X[remaining_idx].astype(np.float64, copy=False)
    n2 = n + 1
    sum_x2 = sum_x[None, :] + Xc
    mu2 = sum_x2 / n2

    xxT = Xc[:, :, None] * Xc[:, None, :]
    cov2 = (sum_xxT[None, :, :] + xxT) / n2 - (mu2[:, :, None] * mu2[:, None, :])

    diag2 = np.einsum("mii,ij->mij", cov2, np.eye(cov2.shape[1], dtype=cov2.dtype))
    cov2a = (1.0 - alpha) * cov2 + alpha * diag2

    dmu = mu2 - mu_full[None, :]
    mu_term = np.einsum("mi,mi->m", dmu, dmu)

    ds = cov2a - sigma_full_alpha[None, :, :]
    cov_term = np.einsum("mij,mij->m", ds, ds)

    L = mu_term + lam * cov_term
    j = int(np.argmin(L))
    return int(remaining_idx[j]), float(L[j])

def sample_by_morphology(morphology_array):
    X = np.asarray(morphology_array)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("morphology_array must be a non-empty 2D array (N, k).")

    X = X.astype(np.float64, copy=False)
    N, k = X.shape

    mean_feat = X.mean(axis=0)
    std_feat = X.std(axis=0, ddof=0)
    std_feat[std_feat == 0.0] = 1.0
    X = (X - mean_feat) / std_feat

    alpha = 0.3
    lam = 0.1
    tol = 1e-2
    max_n = N

    mu = X.mean(axis=0)
    Xm = X - mu
    Sigma = (Xm.T @ Xm) / N
    Sigma_a = _shrink_cov(Sigma, alpha)

    d0 = X - mu[None, :]
    start = int(np.argmin(np.einsum("ni,ni->n", d0, d0)))

    selected = [start]
    in_sel = np.zeros(N, dtype=bool)
    in_sel[start] = True

    sum_x = X[start].copy()
    sum_xxT = np.outer(X[start], X[start])
    n = 1

    cov = sum_xxT / n - np.outer(sum_x / n, sum_x / n)
    L = gaussian_preservation_error(sum_x / n, cov, mu, Sigma, alpha, lam)

    while L > tol and n < max_n:
        print(str(L) + ">" + str(tol))
        print(str(n) + "/" + str(max_n))
        print("-------------")
        remaining = np.flatnonzero(~in_sel)
        best_idx, best_L = greedy_step_add(X, remaining, sum_x, sum_xxT, n, mu, Sigma_a, alpha, lam)
        x = X[best_idx]
        selected.append(best_idx)
        in_sel[best_idx] = True
        sum_x += x
        sum_xxT += np.outer(x, x)
        n += 1

        cov = sum_xxT / n - np.outer(sum_x / n, sum_x / n)
        L = gaussian_preservation_error(sum_x / n, cov, mu, Sigma, alpha, lam)

    selected = np.array(selected, dtype=int)
    return selected, X[selected]

def gaussian_preservation_error(mu_s, sigma_s, mu, sigma, alpha: float, lam: float) -> float:
    mu_s = np.asarray(mu_s, dtype=np.float64)
    mu   = np.asarray(mu,   dtype=np.float64)
    sigma_s = np.asarray(sigma_s, dtype=np.float64)
    sigma   = np.asarray(sigma,   dtype=np.float64)

    dmu = mu_s - mu
    mu_term = dmu @ dmu

    sigma_a   = (1.0 - alpha) * sigma   + alpha * np.diag(np.diag(sigma))
    sigma_s_a = (1.0 - alpha) * sigma_s + alpha * np.diag(np.diag(sigma_s))

    ds = sigma_s_a - sigma_a
    cov_term = np.sum(ds * ds)

    return float(mu_term + lam * cov_term)

if __name__ == "__main__":
    csv_test("cell_morphometry_test_5.csv")