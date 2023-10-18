from __future__ import (  # allows type hint EmbeddingComparison inside EmbeddingComparison class
    annotations,
)

import concurrent.futures
import itertools
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.spatial import procrustes
from sklearn.metrics import fowlkes_mallows_score
from sklearn.preprocessing import normalize

from pandora.custom_errors import PandoraException
from pandora.embedding import MDS, PCA, Embedding


def filter_samples(embedding: Embedding, samples_to_keep: List[str]) -> Embedding:
    """Filters the given Embedding object by removing all samples not contained in samples_to_keep.

    Parameters
    ----------
    embedding : Embedding
        Embedding object to filter.
    samples_to_keep : List[str]
        List of sample IDs to keep.

    Returns
    -------
    Embedding
        new Embedding object containing the data of embedding for all samples in samples_to_keep
    """
    embedding_data = embedding.embedding
    embedding_data = embedding_data.loc[embedding_data.sample_id.isin(samples_to_keep)]

    if isinstance(embedding, PCA):
        return PCA(
            embedding=embedding_data,
            n_components=embedding.n_components,
            explained_variances=embedding.explained_variances,
        )
    elif isinstance(embedding, MDS):
        return MDS(
            embedding=embedding_data,
            n_components=embedding.n_components,
            stress=embedding.stress,
        )
    else:
        raise PandoraException(f"Unrecognized embedding type: {type(embedding)}.")


def _check_sample_clipping(
    before_clipping: Embedding, after_clipping: Embedding
) -> None:
    """Compares the number of samples prior to and after clipping. Will show a warning message in case more than 20% of
    samples were removed, indicating a potential major mismatch between the two PCAs.

    Parameters
    ----------
    before_clipping : PCA
        PCA object prior to sample filtering.
    after_clipping : PCA
        PCA object after sample filtering.

    Returns
    -------
    None

    Warnings
    --------
    UserWarning
        If more than 20% of samples were removed for the comparison. This typically indicates a large number of outliers.
    """
    n_samples_before = before_clipping.embedding.shape[0]
    n_samples_after = after_clipping.embedding.shape[0]

    if n_samples_after <= 0.8 * n_samples_before:
        warnings.warn(
            "More than 20% of samples were removed for the comparison. "
            "Your data appears to have lots of outliers. "
            f"#Samples before/after clipping: {n_samples_before}/{n_samples_after} ",
        )


def _clip_missing_samples_for_comparison(
    comparable: Embedding, reference: Embedding
) -> Tuple[Embedding, Embedding]:
    """Reduces comparable and reference to similar sample IDs to make sure we compare projections for identical samples.

    Parameters
    ----------
    comparable : Embedding
        first Embedding object to compare
    reference : Embedding
        second Embedding object to compare

    Returns
    -------
    comparable : Embedding
        Passed comparable Embedding, but only containing the samples present in both Embeddings (comparable, reference).
    reference : Embedding
        Passed comparable Embedding, but only containing the samples present in both Embeddings (comparable, reference).
    """
    comp_data = comparable.embedding
    ref_data = reference.embedding

    comp_ids = set(comp_data.sample_id)
    ref_ids = set(ref_data.sample_id)

    shared_samples = sorted(comp_ids.intersection(ref_ids))

    comparable_clipped = filter_samples(comparable, shared_samples)
    reference_clipped = filter_samples(reference, shared_samples)

    assert (
        comparable_clipped.embedding_matrix.shape
        == reference_clipped.embedding_matrix.shape
    )
    # Issue a warning if we clip more than 20% of all samples of either Embedding
    # and fail if there are no samples left
    _check_sample_clipping(comparable, comparable_clipped)
    _check_sample_clipping(reference, reference_clipped)

    return comparable_clipped, reference_clipped


def _pad_missing_samples(all_sample_ids: pd.Series, embedding: Embedding) -> Embedding:
    """Pads missing samples with zero-rows."""
    missing_samples = [
        sid for sid in all_sample_ids if sid not in embedding.sample_ids.values
    ]
    if len(missing_samples) == 0:
        return embedding

    _missing_data = [np.zeros(shape=embedding.n_components)] * len(missing_samples)
    embedding_matrix = np.append(embedding.embedding_matrix, _missing_data, axis=0)

    sample_ids = pd.concat(
        [embedding.sample_ids, pd.Series(missing_samples)], ignore_index=True
    )

    _dummy_populations = pd.Series(["_dummy"] * len(missing_samples))
    populations = pd.concat(
        [embedding.populations, _dummy_populations], ignore_index=True
    )

    embedding_matrix = _numpy_to_dataframe(embedding_matrix, sample_ids, populations)

    if isinstance(embedding, PCA):
        return PCA(
            embedding=embedding_matrix,
            n_components=embedding.n_components,
            explained_variances=embedding.explained_variances,
        )
    elif isinstance(embedding, MDS):
        return MDS(
            embedding=embedding_matrix,
            n_components=embedding.n_components,
            stress=embedding.stress,
        )
    else:
        raise PandoraException(f"Unrecognized Embedding type {type(embedding)}.")


def _cluster_stability_for_pair(args):
    i1, embedding1, i2, embedding2, kmeans_k = args
    comparison = EmbeddingComparison(embedding1, embedding2)
    return pd.Series([comparison.compare_clustering(kmeans_k)], index=[(i1, i2)])


def _stability_for_pair(args):
    (i1, embedding1), (i2, embedding2) = args
    comparison = EmbeddingComparison(embedding1, embedding2)
    return pd.Series([comparison.compare()], index=[(i1, i2)])


def _difference_for_pair(args):
    embedding1, embedding2, sample_ids = args
    comp = EmbeddingComparison(embedding1, embedding2)
    embedding1 = _pad_missing_samples(sample_ids, comp.comparable)
    embedding2 = _pad_missing_samples(sample_ids, comp.reference)

    assert (embedding1.sample_ids == embedding2.sample_ids).all()
    return np.linalg.norm(
        embedding1.embedding_matrix - embedding2.embedding_matrix, axis=1
    )


def _get_embedding_norm(args):
    embedding, sample_ids = args
    embedding = _pad_missing_samples(sample_ids, embedding).embedding_matrix
    normalized = normalize(embedding)
    return np.linalg.norm(normalized, axis=1)


class EmbeddingComparison:
    """Class structure for comparing two Embedding results.

    This class provides methods for comparing both Embedding based on all samples,
    for comparing the K-Means clustering results, and for computing sample support values.

    On initialization, comparable and reference are both reduced to contain only samples present in both Embeddings.
    In order to compare the two Embeddings, on initialization Procrustes Analysis is applied transforming
    comparable towards reference. Procrustes Analysis transforms comparable by applying scaling, translation,
    rotation and reflection aiming to match all sample projections as close as possible to the projections in
    reference. Prior to comparing the results, both Embeddings are filtered such that they only contain samples present
    in both Embeddings.

    Note that for comparing Embedding results, the sample IDs are used to ensure the correct comparison of projections.
    If an error occurs during initialization, this is most likely due to incorrect sample IDs.

    Parameters
    ----------
    comparable : Embedding
        Embedding object to compare
    reference : Embedding
        Embedding object to transform comparable towards

    Attributes
    ----------
    comparable : Embedding
        comparable Embedding object after sample filtering and Procrustes Transformation.
    reference : Embedding
        reference Embedding object after sample filtering and Procrustes Transformation.
    sample_ids : pd.Series[str]
        pd.Series containing the sample IDs present in both Embedding objects

    Raises
    ------
    PandoraException
        - If either comparable of reference is not an Embedding object.
        - If comparable and reference are not of the same type (e.g. one is PCA and the other MDS).
    """

    def __init__(self, comparable: Embedding, reference: Embedding):
        if not isinstance(comparable, Embedding) or not isinstance(
            reference, Embedding
        ):
            raise PandoraException(
                f"comparable and reference need to be Embedding objects. "
                f"Instead got {type(comparable)} and {type(reference)}."
            )

        if type(comparable) is not type(reference):
            raise PandoraException(
                f"comparable and reference need to be of the same Embedding type. "
                f"Instead got {type(comparable)} and {type(reference)}."
            )

        self.comparable, self.reference, self.disparity = match_and_transform(
            comparable=comparable, reference=reference
        )
        self.sample_ids = self.comparable.sample_ids

    def compare(self) -> float:
        """Computes the Pandora stability between `self.comparable` to `self.reference` using Procrustes Analysis.

        Returns
        -------
        float
            Similarity score on a scale of 0 (entirely different) to 1 (identical) measuring the similarity of
            `self.comparable` and `self.reference`.
        """
        similarity = np.sqrt(1 - self.disparity)
        return similarity

    def compare_clustering(self, kmeans_k: int = None) -> float:
        """Computes the Pandora cluster stability between self.comparable and self.reference.

        Compares the assigned cluster labels based on K-Means clustering on both embeddings.

        Parameters
        ----------
        kmeans_k : int, default=None
            Number k of clusters to use for K-Means clustering.
            If not set, the optimal number of clusters is determined automatically using self.reference.

        Returns
        -------
        float
            The Fowlkes-Mallow score of Cluster similarity between the clustering results of self.reference and
            self.comparable. The score ranges from 0 (entirely distinct) to 1 (identical).
        """
        if kmeans_k is None:
            # we are comparing self to other -> use other as ground truth
            # thus, we determine the number of clusters using other
            kmeans_k = self.reference.get_optimal_kmeans_k()

        # since we are only comparing the assigned cluster labels, we don't need to transform self prior to comparing
        comp_kmeans = self.comparable.cluster(kmeans_k=kmeans_k)
        ref_kmeans = self.reference.cluster(kmeans_k=kmeans_k)

        comp_cluster_labels = comp_kmeans.predict(self.comparable.embedding_matrix)
        ref_cluster_labels = ref_kmeans.predict(self.reference.embedding_matrix)

        return fowlkes_mallows_score(ref_cluster_labels, comp_cluster_labels)


class BatchEmbeddingComparison:
    """Class structure for comparing three or more Embedding results. All comparisons are conducted pairwise for all
    unique pairs of embeddings.

    Embedding objects must all be of the same type, allowed types are PCA and MDS.

    This class provides methods for comparing both Embedding based on all samples,
    for comparing the K-Means clustering results, and for computing sample support values.

    BatchEmbeddingComparison makes use of pairwise EmbeddingComparison objects for comparing results pairwise.

    Parameters
    ----------
    embeddings : List[Embedding]
        List of embeddings to compare

    Attributes
    ----------
    embeddings : List[Embedding]
        List of embeddings to compare

    Raises
    ------
    PandoraException
        - If not all embeddings are of the same type.
        - If not all embeddings are either PCA or MDS objects.
        - If less than three embeddings are passed.
    """

    def __init__(self, embeddings: List[Embedding]):
        types = set(type(e) for e in embeddings)
        if len(types) > 1:
            raise PandoraException("All Embeddings need to be of identical types.")

        if types.pop() not in [PCA, MDS]:
            raise PandoraException("All Embeddings must be PCA or MDS embeddings.")

        if len(embeddings) < 3:
            raise PandoraException(
                "BatchEmbeddingComparison of Embeddings makes only sense for three or more embeddings."
                "For two use the EmbeddingComparison class."
            )
        self.embeddings = embeddings

    def get_pairwise_stabilities(self, threads: Optional[int] = None) -> pd.Series:
        """Computes the pairwise Pandora stability scores for all unique pairs of `self.embedding` and stores them in a
        pandas Series.

        Parameters
        ----------
        threads : int, default=None
            Number of threads to use for the computation. Default is to use all available system threads.

        Returns
        -------
        pd.Series
            Pandas Series containing the pairwise stability scores for all unique pairs of
            self.embeddings. The resulting Series is named pandora_stability and has the indices of the
            pairwise comparisons as index. So a result looks e.g. like this::

                (0, 1)    0.93
                (0, 2)    0.79
                (1, 2)    0.71
                Name: pandora_stability, dtype: float64

            Each value is between 0 and 1 with higher values indicating a higher stability.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as pool:
            pairwise_stabilities = pool.map(
                _stability_for_pair,
                itertools.combinations(enumerate(self.embeddings), r=2),
            )

        pairwise_stabilities = pd.concat(pairwise_stabilities)
        pairwise_stabilities.name = "pandora_stability"
        return pairwise_stabilities

    def compare(self, threads: Optional[int] = None) -> float:
        """Compares all embeddings pairwise and returns the average of the resulting pairwise Pandora stability scores.

        See EmbeddingComparison::compare for more details on how the pairwise Pandora stability is computed.

        Parameters
        ----------
        threads: int, default=None
            Number of threads to use for the computation. Default is to use all available system threads.

        Returns
        -------
        float
            Average of pairwise Pandora stability scores. This value is between 0 and 1 with higher values indicating
            a higher stability.
        """
        return self.get_pairwise_stabilities(threads).mean()

    def get_pairwise_cluster_stabilities(
        self, kmeans_k: int, threads: Optional[int] = None
    ) -> pd.DataFrame:
        """Computes the pairwise Pandora cluster stability scores for all unique pairs of `self.embeddin`g and stores
        them in a pandas Series.

        Parameters
        ----------
        kmeans_k : int
            Number k of clusters to use for K-Means clustering.
        threads : int, default=None
            Number of threads to use for the computation. Default is to use all available system threads.

        Returns
        -------
        pd.Series
            Pandas Series containing the pairwise cluster stability scores for all unique pairs of
            self.embeddings. The resulting Series is named pandora_cluster_stability and has the indices of the
            pairwise comparisons as index. So a result looks e.g. like this:
            (0, 1)    0.93
            (0, 2)    0.79
            (1, 2)    0.71
            Name: pandora_cluster_stability, dtype: float64
            Each value is between 0 and 1 with higher values indicating a higher stability.
        """
        args = [
            (i1, embedding1, i2, embedding2, kmeans_k)
            for (i1, embedding1), (i2, embedding2) in itertools.combinations(
                enumerate(self.embeddings), r=2
            )
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as pool:
            pairwise_cluster_stabilities = pool.map(_cluster_stability_for_pair, args)

        pairwise_cluster_stabilities = pd.concat(pairwise_cluster_stabilities)
        pairwise_cluster_stabilities.name = "pandora_cluster_stability"
        return pairwise_cluster_stabilities

    def compare_clustering(self, kmeans_k: int, threads: Optional[int] = None) -> float:
        """Compares all embeddings pairwise and returns the average of the resulting pairwise Pandora cluster stability
        scores.

        See EmbeddingComparison::compare_clustering for more details on how the pairwise Pandora cluster stability is computed.

        Parameters
        ----------
        kmeans_k : int
            Number k of clusters to use for K-Means clustering.
        threads : int, default=None
            Number of threads to use for the computation. Default is to use all available system threads.

        Returns
        -------
        float
            Average of pairwise Pandora cluster stability scores. This value is between 0 and 1 with higher values
            indicating a higher stability.
        """
        return self.get_pairwise_cluster_stabilities(kmeans_k, threads).mean()

    def _get_pairwise_difference(
        self, sample_ids: pd.Series, threads: Optional[int] = None
    ):
        args = [
            (embedding1, embedding2, sample_ids)
            for embedding1, embedding2 in itertools.permutations(self.embeddings, r=2)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as pool:
            diffs = pool.map(_difference_for_pair, args)
        return diffs

    def get_sample_support_values(self, threads: Optional[int] = None) -> pd.Series:
        """Computes the sample support value for each sample respective all self.embeddings.

        The sample support value per sample is computed as the `1 - dispertion` across all embeddings where dispertion
        is computed using the Gini Coefficient.
        The support values are computed for all samples in the union of all sample IDs of all self.embeddings.

        Parameters
        ----------
        threads : int, default=None
            Number of threads to use for the computation. Default is to use all available system threads.

        Returns
        -------
        pd.Series
            Pandas Series containing the support values for all samples across all pairwise embedding comparisons.
            Each row corresponds to a sample, with the sample IDs as indices and the PSV as value. The name of the series
            is set to `PSV`.
        """
        sample_ids_superset = set(
            [sid for embedding in self.embeddings for sid in embedding.sample_ids]
        )
        sample_ids_superset = pd.Series(list(sample_ids_superset)).sort_values()

        numerator = np.sum(
            self._get_pairwise_difference(sample_ids_superset, threads), axis=0
        )

        args = [(embedding, sample_ids_superset) for embedding in self.embeddings]
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as pool:
            embedding_norms = pool.map(_get_embedding_norm, args)

        denominator = 2 * len(self.embeddings) * np.sum(embedding_norms, axis=0) + 1e-6
        gini_coefficients = pd.Series(
            numerator / denominator, index=sample_ids_superset.values, name="PSV"
        ).sort_index()
        return 1 - gini_coefficients


def _numpy_to_dataframe(
    embedding_matrix: npt.NDArray[float],
    sample_ids: pd.Series[str],
    populations: pd.Series[str],
):
    """Transforms a numpy ndarray to a pandas Dataframe as required for initializing an Embedding object.

    Parameters
    ----------
    embedding_matrix : npt.NDArray[float]
        Numpy ndarray containing the Embedding results (PC vectors) for all samples.
    sample_ids : pd.Series[str]
        Pandas Series containing the sample IDs corresponding to the embedding_matrix.
    populations : pd.Series[str]
        Pandas Series containing the populations corresponding to the sample_ids.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing all required columns to initialize a Embedding object
        (sample_id, population, D{i} for i in range(embedding_matrix.shape[1]))
    """
    if embedding_matrix.ndim != 2:
        raise PandoraException(
            f"Numpy embedding matrix must be two dimensional. Passed data has {embedding_matrix.ndim} dimensions."
        )

    embedding_data = pd.DataFrame(
        embedding_matrix, columns=[f"D{i}" for i in range(embedding_matrix.shape[1])]
    )

    if sample_ids.shape[0] != embedding_data.shape[0]:
        raise PandoraException(
            f"One sample ID required for each sample. Got {len(sample_ids)} IDs, "
            f"but embedding_data has {embedding_data.shape[0]} sample_ids."
        )

    embedding_data["sample_id"] = sample_ids.values

    if populations.shape[0] != embedding_data.shape[0]:
        raise PandoraException(
            f"One population required for each sample. Got {len(populations)} populations, "
            f"but embedding_data has {embedding_data.shape[0]} sample_ids."
        )

    embedding_data["population"] = populations.values
    return embedding_data


def match_and_transform(
    comparable: Embedding, reference: Embedding
) -> Tuple[Embedding, Embedding, float]:
    """Uses Procrustes Analysis to find a transformation matrix that most closely matches comparable to reference. and
    transforms comparable.

    Parameters
    ----------
    comparable : Embedding
        The Embedding that should be transformed
    reference : Embedding
        The Embedding that comparable should be transformed towards

    Returns
    -------
    transformed_comparable : Embedding
        Transformed comparable Embedding, created by matching `comparable` to `reference` as close as possible using
        Procrustes Analysis.
    standardized_reference : Embedding
        Standardized reference Embedding, `reference` is standardized during the matching procedure by Procrustes Analysis.
    disparity : float
        The sum of squared distances between the transformed comparable and transformed reference Embeddings.

    Raises
    ------
    PandoraException
        - Mismatch in sample IDs between comparable and reference (identical sample IDs required for comparison).
        - Mismatch in number of samples of PCs in comparable and reference.
        - No samples left after clipping. This is most likely caused by incorrect annotations of sample IDs.
        - Comparable and reference are of different types. Both Embeddigs need to be either PCA or MDS objects.
    """
    comparable, reference = _clip_missing_samples_for_comparison(comparable, reference)

    if not all(comparable.sample_ids == reference.sample_ids):
        raise PandoraException(
            "Sample IDS between reference and comparable don't match but is required for comparing PCA results. "
        )

    comp_data = comparable.embedding_matrix
    ref_data = reference.embedding_matrix

    if comp_data.shape != ref_data.shape:
        raise PandoraException(
            f"Number of samples or PCs in comparable and reference do not match. "
            f"Got {comp_data.shape} and {ref_data.shape} respectively."
        )

    if comp_data.shape[0] == 0:
        raise PandoraException(
            "No samples left for comparison after clipping. "
            "Make sure all sample IDs are correctly annotated"
        )

    standardized_reference, transformed_comparable, disparity = procrustes(
        ref_data, comp_data
    )

    # normalize the data prior to comparison
    standardized_reference = normalize(standardized_reference)
    transformed_comparable = normalize(transformed_comparable)

    standardized_reference = _numpy_to_dataframe(
        standardized_reference,
        reference.sample_ids,
        reference.populations,
    )

    transformed_comparable = _numpy_to_dataframe(
        transformed_comparable,
        comparable.sample_ids,
        comparable.populations,
    )

    if isinstance(reference, PCA) and isinstance(comparable, PCA):
        standardized_reference = PCA(
            embedding=standardized_reference,
            n_components=reference.n_components,
            explained_variances=reference.explained_variances,
        )

        transformed_comparable = PCA(
            embedding=transformed_comparable,
            n_components=comparable.n_components,
            explained_variances=comparable.explained_variances,
        )

    elif isinstance(reference, MDS) and isinstance(comparable, MDS):
        standardized_reference = MDS(
            embedding=standardized_reference,
            n_components=reference.n_components,
            stress=reference.stress,
        )

        transformed_comparable = MDS(
            embedding=transformed_comparable,
            n_components=comparable.n_components,
            stress=comparable.stress,
        )

    else:
        raise PandoraException(
            "comparable and reference need to be of type PCA or MDS."
        )

    if not all(standardized_reference.sample_ids == transformed_comparable.sample_ids):
        raise PandoraException(
            "Sample IDS between reference and comparable don't match but is required for comparing PCA results. "
        )

    return standardized_reference, transformed_comparable, disparity
