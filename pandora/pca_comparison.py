from __future__ import (
    annotations,
)  # allows type hint PCAComparison inside PCAComparison class

import warnings

import pandas as pd
from scipy.spatial import procrustes
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import euclidean_distances

from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.pca import PCA


def _correct_missing(pca: PCA, samples_in_both):
    pca_data = pca.pca_data
    pca_data = pca_data.loc[pca_data.sample_id.isin(samples_in_both)]

    return PCA(
        pca_data=pca_data, explained_variances=pca.explained_variances, n_pcs=pca.n_pcs
    )


def _check_sample_clipping(before_clipping: PCA, after_clipping: PCA) -> None:
    n_samples_before = before_clipping.pca_data.shape[0]
    n_samples_after = after_clipping.pca_data.shape[0]

    if n_samples_after == 0:
        raise RuntimeWarning("All samples were removed for the comparison.")

    if n_samples_after <= 0.8 * n_samples_before:
        warnings.warn(
            "More than 20% of samples were removed for the comparison. "
            "Your data appears to have lots of outliers. "
            f"#Samples before/after clipping: {n_samples_before}/{n_samples_after} ",
        )


def _clip_missing_samples_for_comparison(
    comparable: PCA, reference: PCA
) -> Tuple[PCA, PCA]:
    comp_data = comparable.pca_data
    ref_data = reference.pca_data

    comp_ids = set(comp_data.sample_id)
    ref_ids = set(ref_data.sample_id)

    shared_samples = sorted(comp_ids.intersection(ref_ids))

    comparable_clipped = _correct_missing(comparable, shared_samples)
    reference_clipped = _correct_missing(reference, shared_samples)

    assert comparable_clipped.pc_vectors.shape == reference_clipped.pc_vectors.shape
    # Issue a warning if we clip more than 20% of all samples of either PCA
    # and fail if there are no samples lef
    _check_sample_clipping(comparable, comparable_clipped)
    _check_sample_clipping(reference, reference_clipped)

    return comparable_clipped, reference_clipped


class PCAComparison:
    def __init__(self, comparable: PCA, reference: PCA):
        # first we transform comparable towards reference such that we can compare the PCAs
        self.comparable, self.reference, self.disparity = match_and_transform(
            comparable=comparable, reference=reference
        )
        self.sample_ids = self.comparable.pca_data.sample_id

    def compare(self) -> float:
        """
        Compare self to other by transforming self towards other and then computing the samplewise cosine similarity.
        Returns the average and standard deviation. The resulting similarity is on a scale of 0 to 1, with 1 meaning
        self and other are identical.
        TODO: nope wir nehmen jetzt doch procrustes und die similarity von procrustes direkt

        Args:
            other (PCA): PCA object to compare self to.

        Returns:
            float: Similarity as average cosine similarity per sample PC-vector in self and other.
        """
        similarity = np.sqrt(1 - self.disparity)

        return similarity

    def compare_clustering(self, kmeans_k: int = None) -> float:
        """
        Compare self clustering to other clustering using other as ground truth.

        Args:
            other (PCA): PCA object to compare self to.
            kmeans_k (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.

        Returns:
            float: The Fowlkes-Mallow score of Cluster similarity between the clusters of self and other
        """
        if kmeans_k is None:
            # we are comparing self to other -> use other as ground truth
            # thus, we determine the number of clusters using other
            kmeans_k = self.reference.get_optimal_kmeans_k()

        # since we are only comparing the assigned cluster labels, we don't need to transform self prior to comparing
        comp_kmeans = self.comparable.cluster(kmeans_k=kmeans_k)
        ref_kmeans = self.reference.cluster(kmeans_k=kmeans_k)

        comp_cluster_labels = comp_kmeans.predict(self.comparable.pc_vectors)
        ref_cluster_labels = ref_kmeans.predict(self.reference.pc_vectors)

        return fowlkes_mallows_score(ref_cluster_labels, comp_cluster_labels)

    def _get_sample_distances(self) -> pd.Series:
        # make sure we are comparing the correct PC-vectors in the following
        assert np.all(
            self.comparable.pca_data.sample_id == self.reference.pca_data.sample_id
        )

        sample_distances = euclidean_distances(
            self.reference.pc_vectors, self.comparable.pc_vectors
        ).diagonal()
        return pd.Series(sample_distances)

    def get_sample_support_values(self) -> pd.DataFrame:
        sample_distances = self._get_sample_distances()
        support_values = 1 / (1 + sample_distances)
        return pd.DataFrame(
            data={"support": support_values.values}, index=self.sample_ids
        )

    def detect_rogue_samples(
        self, support_value_rogue_cutoff: float = 0.5
    ) -> pd.DataFrame:
        """
        TODO: Docstring updaten
        Returns a list of sample IDs that are considered rogue samples when comparing self.comparable to self.reference.
        A sample is considered rogue if the euclidean distance between its PC vectors in self and other
        is larger than the rogue_cutoff-percentile of pairwise PC vector distances
        """
        support_values = self.get_sample_support_values()

        rogue = support_values.loc[lambda x: (x.support < support_value_rogue_cutoff)]

        return rogue


def _numpy_to_pca_dataframe(
    pc_vectors: npt.NDArray[float],
    sample_ids: pd.Series[str],
    populations: pd.Series[str],
):
    if pc_vectors.ndim != 2:
        raise PandoraException(
            f"Numpy PCA data must be two dimensional. Passed data has {pc_vectors.ndim} dimensions."
        )

    pca_data = pd.DataFrame(
        pc_vectors, columns=[f"PC{i}" for i in range(pc_vectors.shape[1])]
    )

    if sample_ids.shape[0] != pca_data.shape[0]:
        raise PandoraException(
            f"One sample ID required for each sample. Got {len(sample_ids)} IDs, "
            f"but pca_data has {pca_data.shape[0]} samples."
        )

    pca_data["sample_id"] = sample_ids.values

    if populations.shape[0] != pca_data.shape[0]:
        raise PandoraException(
            f"One population required for each sample. Got {len(populations)} populations, "
            f"but pca_data has {pca_data.shape[0]} samples."
        )

    pca_data["population"] = populations.values
    return pca_data


def match_and_transform(comparable: PCA, reference: PCA) -> Tuple[PCA, PCA, float]:
    """
    Finds a transformation matrix that most closely matches comparable to reference and transforms comparable.

    Args:
        comparable (PCA): The PCA that should be transformed
        reference (PCA): The PCA that comparable should be transformed towards

    Returns:
        Tuple[PCA, PCA]: Two new PCA objects, the first one is the transformed comparable and the second one is the standardized reference.
            In all downstream comparisons or pairwise plotting, use these PCA objects.
    """
    comparable, reference = _clip_missing_samples_for_comparison(comparable, reference)

    assert all(comparable.pca_data.sample_id == reference.pca_data.sample_id)

    comp_data = comparable.pc_vectors
    ref_data = reference.pc_vectors

    # TODO: reorder PCs (if we find a dataset where this is needed...don't want to blindly implement something)
    # check if the number of samples match for now
    assert comp_data.shape == ref_data.shape

    standardized_reference, transformed_comparable, disparity = procrustes(
        ref_data, comp_data
    )

    standardized_reference = _numpy_to_pca_dataframe(
        standardized_reference,
        reference.pca_data.sample_id,
        reference.pca_data.population,
    )

    standardized_reference = PCA(
        pca_data=standardized_reference,
        explained_variances=reference.explained_variances,
        n_pcs=reference.n_pcs,
    )

    transformed_comparable = _numpy_to_pca_dataframe(
        transformed_comparable,
        comparable.pca_data.sample_id,
        comparable.pca_data.population,
    )
    transformed_comparable = PCA(
        pca_data=transformed_comparable,
        explained_variances=comparable.explained_variances,
        n_pcs=comparable.n_pcs,
    )

    assert all(
        standardized_reference.pca_data.sample_id
        == transformed_comparable.pca_data.sample_id
    )

    return standardized_reference, transformed_comparable, disparity
