import numpy as np
import pandas as pd


def load_morphology_csv(filepath: str) -> np.ndarray:
    X = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=1,
        dtype=np.float64,
    )

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("CSV must contain a non-empty numeric 2D array.")

    if np.isnan(X).any():
        raise ValueError("CSV contains non-numeric or missing values beyond the header.")

    return X


def generate_random_clusters_from_file(
    filepath: str,
    min_cluster_size: int = 1,
    max_cluster_size: int = 5,
    seed: int | None = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive.")

    if max_cluster_size < min_cluster_size:
        raise ValueError("max_cluster_size must be >= min_cluster_size.")

    X = load_morphology_csv(filepath)
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(N)

    clusters = []
    pos = 0

    while pos < N:
        remaining = N - pos

        if remaining <= max_cluster_size:
            cluster_size = remaining
        else:
            cluster_size = int(rng.integers(min_cluster_size, max_cluster_size + 1))

        cluster = shuffled_indices[pos : pos + cluster_size]
        clusters.append(np.sort(cluster))
        pos += cluster_size

    return X, clusters


def clusters_to_dataframe(clusters: list[np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame([cluster.tolist() for cluster in clusters])


def _shrink_cov(cov: np.ndarray, alpha: float) -> np.ndarray:
    diagonal_cov = np.diag(np.diag(cov))
    return (1.0 - alpha) * cov + alpha * diagonal_cov


def gaussian_preservation_error(
    mu_s: np.ndarray,
    sigma_s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    lam: float,
) -> float:
    mu_s = np.asarray(mu_s, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma_s = np.asarray(sigma_s, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    dmu = mu_s - mu
    mu_term = dmu @ dmu

    sigma_s_alpha = _shrink_cov(sigma_s, alpha)
    sigma_alpha = _shrink_cov(sigma, alpha)

    ds = sigma_s_alpha - sigma_alpha
    cov_term = np.sum(ds * ds)

    return float(mu_term + lam * cov_term)


def normalize_clusters(clusters, N: int) -> list[np.ndarray]:
    if isinstance(clusters, pd.DataFrame):
        cluster_list = []

        for _, row in clusters.iterrows():
            values = row.dropna().to_numpy(dtype=int)
            cluster_list.append(values)
    else:
        cluster_list = [np.asarray(cluster, dtype=int) for cluster in clusters]

    seen = set()
    clean_clusters = []

    for cluster in cluster_list:
        if cluster.ndim != 1:
            raise ValueError("Each cluster must be a 1D list or array of row indices.")

        if len(cluster) == 0:
            continue

        if np.any(cluster < 0) or np.any(cluster >= N):
            raise ValueError(
                f"Cluster contains indices outside valid range 0..{N - 1}: {cluster}"
            )

        for idx in cluster:
            idx = int(idx)

            if idx in seen:
                raise ValueError(f"Index {idx} appears in more than one cluster.")

            seen.add(idx)

        clean_clusters.append(cluster.astype(int))

    if len(clean_clusters) == 0:
        raise ValueError("At least one non-empty cluster is required.")

    return clean_clusters


def precompute_cluster_stats(
    X: np.ndarray,
    clusters: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = X.shape[1]
    num_clusters = len(clusters)

    cluster_sizes = np.empty(num_clusters, dtype=int)
    cluster_sums = np.empty((num_clusters, k), dtype=np.float64)
    cluster_sums_xxT = np.empty((num_clusters, k, k), dtype=np.float64)

    for i, idx in enumerate(clusters):
        X_cluster = X[idx]

        cluster_sizes[i] = len(idx)
        cluster_sums[i] = X_cluster.sum(axis=0)
        cluster_sums_xxT[i] = X_cluster.T @ X_cluster

    return cluster_sizes, cluster_sums, cluster_sums_xxT


def greedy_step_add_cluster(
    remaining_cluster_idx: np.ndarray,
    cluster_sizes: np.ndarray,
    cluster_sums: np.ndarray,
    cluster_sums_xxT: np.ndarray,
    sum_x: np.ndarray,
    sum_xxT: np.ndarray,
    n: int,
    mu_full: np.ndarray,
    sigma_full_alpha: np.ndarray,
    alpha: float,
    lam: float,
) -> tuple[int, float]:
    sizes = cluster_sizes[remaining_cluster_idx]
    sums = cluster_sums[remaining_cluster_idx]
    sums_xxT = cluster_sums_xxT[remaining_cluster_idx]

    n2 = n + sizes

    sum_x2 = sum_x[None, :] + sums
    mu2 = sum_x2 / n2[:, None]

    cov2 = (
        (sum_xxT[None, :, :] + sums_xxT) / n2[:, None, None]
        - mu2[:, :, None] * mu2[:, None, :]
    )

    cov2_alpha = (1.0 - alpha) * cov2
    diagonal_idx = np.arange(cov2.shape[1])
    cov2_alpha[:, diagonal_idx, diagonal_idx] += alpha * cov2[
        :, diagonal_idx, diagonal_idx
    ]

    dmu = mu2 - mu_full[None, :]
    mu_term = np.einsum("mi,mi->m", dmu, dmu)

    ds = cov2_alpha - sigma_full_alpha[None, :, :]
    cov_term = np.einsum("mij,mij->m", ds, ds)

    losses = mu_term + lam * cov_term

    best_local_idx = int(np.argmin(losses))
    best_cluster_idx = int(remaining_cluster_idx[best_local_idx])
    best_loss = float(losses[best_local_idx])

    return best_cluster_idx, best_loss


def sample_by_morphology_clustered(
    morphology_array: np.ndarray,
    clusters,
    alpha: float = 0.3,
    lam: float = 0.1,
    tol: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(morphology_array)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("morphology_array must be a non-empty 2D array of shape (N, k).")

    X = X.astype(np.float64, copy=False)
    N, _ = X.shape

    clusters = normalize_clusters(clusters, N)

    mean_feat = X.mean(axis=0)
    std_feat = X.std(axis=0, ddof=0)
    std_feat[std_feat == 0.0] = 1.0
    X = (X - mean_feat) / std_feat

    mu = X.mean(axis=0)
    X_centered = X - mu
    sigma = (X_centered.T @ X_centered) / N
    sigma_alpha = _shrink_cov(sigma, alpha)

    cluster_sizes, cluster_sums, cluster_sums_xxT = precompute_cluster_stats(
        X,
        clusters,
    )

    num_clusters = len(clusters)
    initial_losses = np.empty(num_clusters, dtype=np.float64)

    for cluster_idx in range(num_clusters):
        n0 = int(cluster_sizes[cluster_idx])
        sum_x0 = cluster_sums[cluster_idx]
        sum_xxT0 = cluster_sums_xxT[cluster_idx]

        mu0 = sum_x0 / n0
        cov0 = sum_xxT0 / n0 - np.outer(mu0, mu0)

        initial_losses[cluster_idx] = gaussian_preservation_error(
            mu0,
            cov0,
            mu,
            sigma,
            alpha,
            lam,
        )

    start_cluster = int(np.argmin(initial_losses))

    selected_clusters = [start_cluster]
    in_selected_cluster = np.zeros(num_clusters, dtype=bool)
    in_selected_cluster[start_cluster] = True

    sum_x = cluster_sums[start_cluster].copy()
    sum_xxT = cluster_sums_xxT[start_cluster].copy()
    n = int(cluster_sizes[start_cluster])

    cov = sum_xxT / n - np.outer(sum_x / n, sum_x / n)
    loss = gaussian_preservation_error(
        sum_x / n,
        cov,
        mu,
        sigma,
        alpha,
        lam,
    )

    while loss > tol and not np.all(in_selected_cluster):
        remaining_clusters = np.flatnonzero(~in_selected_cluster)

        best_cluster_idx, best_loss = greedy_step_add_cluster(
            remaining_clusters,
            cluster_sizes,
            cluster_sums,
            cluster_sums_xxT,
            sum_x,
            sum_xxT,
            n,
            mu,
            sigma_alpha,
            alpha,
            lam,
        )

        selected_clusters.append(best_cluster_idx)
        in_selected_cluster[best_cluster_idx] = True

        sum_x += cluster_sums[best_cluster_idx]
        sum_xxT += cluster_sums_xxT[best_cluster_idx]
        n += int(cluster_sizes[best_cluster_idx])

        cov = sum_xxT / n - np.outer(sum_x / n, sum_x / n)
        loss = gaussian_preservation_error(
            sum_x / n,
            cov,
            mu,
            sigma,
            alpha,
            lam,
        )

    selected_indices = np.concatenate(
        [clusters[cluster_idx] for cluster_idx in selected_clusters]
    )

    return (
        selected_indices.astype(int),
        np.array(selected_clusters, dtype=int),
        float(loss),
    )


def csv_test_clustered(
    filepath: str,
    min_cluster_size: int = 1,
    max_cluster_size: int = 5,
    seed: int | None = 42,
) -> None:
    X, clusters = generate_random_clusters_from_file(
        filepath,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        seed=seed,
    )

    clusters_df = clusters_to_dataframe(clusters)

    selected_indices, selected_clusters, final_loss = sample_by_morphology_clustered(
        X,
        clusters_df,
    )

    print("Generated clusters:")
    print(clusters_df)

    print("\nSelected cluster IDs:")
    print(selected_clusters)

    print("\nSelected row indices:")
    print(selected_indices)

    print("\nFinal loss:")
    print(final_loss)

    print("\nNumber of selected rows:")
    print(len(selected_indices))

    print("\nNumber of selected clusters:")
    print(len(selected_clusters))


if __name__ == "__main__":
    csv_test_clustered(
        "cell_morphometry_test_5.csv",
        min_cluster_size=1,
        max_cluster_size=5,
        seed=42,
    )