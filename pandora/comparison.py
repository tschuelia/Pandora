import numpy as np
import pandas as pd

from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score, fowlkes_mallows_score

from pandora.custom_types import *
from pandora.logger import logger, fmt_message


def prepare_pca_for_comparison(pca1: pd.DataFrame, pca2: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if pca1.shape[0] != pca2.shape[0]:
        raise ValueError("Mismatch in PCA size: PCA1 and PCA2 need to have the same number of rows.")

    pca1 = pca1.sort_values(by="sample_id").reset_index(drop=True)
    pca2 = pca2.sort_values(by="sample_id").reset_index(drop=True)

    pca1 = pca1[[c for c in pca1.columns if "PC" in c]].to_numpy()
    pca2 = pca2[[c for c in pca1.columns if "PC" in c]].to_numpy()

    n_pcs1 = pca1.shape[1]
    n_pcs2 = pca2.shape[1]

    n_pcs_target = min(n_pcs1, n_pcs2)

    if n_pcs1 != n_pcs2:
        logger.warning(
            fmt_message(
                f"Mismatch in number of PCs: PCA 1 has {n_pcs1} PCs, PCA 2 has {n_pcs2} PCs."
                f"Will only compare the first {n_pcs_target} PCs."
            )
        )
        pca1 = pca1[:, :n_pcs_target]
        pca2 = pca2[:, :n_pcs_target]

    return pca1, pca2


def match_pcas(pca1: pd.DataFrame, pca2: pd.DataFrame, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: reorder PCs (if we find a dataset where this is needed...don't want to blindly implement something)
    pca1, pca2 = prepare_pca_for_comparison(pca1, pca2)

    if normalize:
        pca1 = pca1 / np.linalg.norm(pca1)
        pca2 = pca2 / np.linalg.norm(pca2)

    transformation, _ = orthogonal_procrustes(pca1, pca2)
    transformed_pca2 = pca2 @ transformation

    return pca1, transformed_pca2, transformation


def find_optimal_number_of_clusters(pca: np.ndarray):
    best_k = -1
    best_score = -1
    for k in range(3, 50):
        # TODO: what is the maximum k that is reasonable>
        kmeans = KMeans(random_state=42, n_clusters=k)
        kmeans.fit(pca)
        score = silhouette_score(pca, kmeans.labels_)
        best_k = k if score > best_score else best_k
        best_score = max(score, best_score)

    return best_k


def compare_clustering(pca1: pd.DataFrame, pca2: pd.DataFrame) -> Dict:
    pca1, transformed_pca2, _ = match_pcas(pca1, pca2)

    n_clusters = find_optimal_number_of_clusters(pca1)

    pca1_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pca1_kmeans.fit(pca1)

    pca2_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pca2_kmeans.fit(transformed_pca2)

    true_clusters = pca1_kmeans.predict(pca1)
    predicted_clusters = pca2_kmeans.predict(transformed_pca2)

    scores = {
        "rand_score": rand_score(true_clusters, predicted_clusters),
        "adjusted_rand_score": adjusted_rand_score(true_clusters, predicted_clusters),
        "v_measure_score": v_measure_score(true_clusters, predicted_clusters),
        "adjusted_mutual_info_score": adjusted_mutual_info_score(true_clusters, predicted_clusters),
        "fowlkes_mallows_score": fowlkes_mallows_score(true_clusters, predicted_clusters)
    }

    return scores
