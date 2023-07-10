from __future__ import (
    annotations,
)  # allows type hint DimRedComparison inside DimRedComparison class

import warnings

import pandas as pd
from scipy.spatial import procrustes
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.dimensionality_reduction import PCA, MDS, DimRedBase


def filter_samples(dimred: DimRedBase, samples_to_keep: List[str]) -> DimRedBase:
    """
    Filters the given DimRedBase object by removing all samples not contained in samples_to_keep

    Args:
        dimred (DimRedBase): DimRedBase object to filter.
        samples_to_keep (List[str]): List of sample IDs to keep.

    Returns: new DimRedBase object containing the data of pca for all samples in samples_to_keep

    """
    embedding = dimred.embedding
    embedding = embedding.loc[embedding.sample_id.isin(samples_to_keep)]

    if isinstance(dimred, PCA):
        return PCA(embedding=embedding, n_components=dimred.n_components, explained_variances=dimred.explained_variances)
    elif isinstance(dimred, MDS):
        return MDS(embedding=embedding, n_components=dimred.n_components, stress=dimred.stress)
    else:
        raise PandoraException(f"Unrecognized dimred type: {type(dimred)}.")


def _check_sample_clipping(before_clipping: DimRedBase, after_clipping: DimRedBase) -> None:
    """
    Compares the number of samples prior to and after clipping. Will show a warning message in case more
    than 20% of samples were removed, indicating a potential major mismatch between the two PCAs.

    Args:
        before_clipping (PCA): PCA object prior to sample filtering.
        after_clipping (PCA): PCA object after sample filtering.

    Returns: None

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
    comparable: DimRedBase, reference: DimRedBase
) -> Tuple[DimRedBase, DimRedBase]:
    """
    Reduces comparable and reference to similar sample IDs to make sure we compare projections for identical samples.

    Args:
        comparable (DimRedBase): first DimRedBase object to compare
        reference (DimRedBase): second DimRedBase object to compare

    Returns:
        (DimRedBase, DimRedBase): Comparable and reference DimRedBases containing only the samples present in both DimRedBases.

    """
    comp_data = comparable.embedding
    ref_data = reference.embedding

    comp_ids = set(comp_data.sample_id)
    ref_ids = set(ref_data.sample_id)

    shared_samples = sorted(comp_ids.intersection(ref_ids))

    comparable_clipped = filter_samples(comparable, shared_samples)
    reference_clipped = filter_samples(reference, shared_samples)

    assert comparable_clipped.embedding_matrix.shape == reference_clipped.embedding_matrix.shape
    # Issue a warning if we clip more than 20% of all samples of either DimRedBase
    # and fail if there are no samples lef
    _check_sample_clipping(comparable, comparable_clipped)
    _check_sample_clipping(reference, reference_clipped)

    return comparable_clipped, reference_clipped


class DimRedComparison:
    """Class structure for comparing two DimRedBase results.

    This class provides methods for comparing both DimRedBases based on all samples,
    for comparing the K-Means clustering results, and for computing sample support values.

    Prior to comparing the results, both DimRedBases are filtered such that they only contain samples present in both DimRedBases.

    Note that for comparing DimRedBase results, the sample IDs are used to ensure the correct comparison of projections.
    If an error occurs during initialization, this is most likely due to incorrect sample IDs.

    Attributes:
        comparable (DimRedBase): comparable DimRedBase object after sample filtering and Procrustes Transformation.
        reference (DimRedBase): reference DimRedBase object after sample filtering and Procrustes Transformation.
        sample_ids (pd.Series[str]): pd.Series containing the sample IDs present in both DimRedBase objects
    """

    def __init__(self, comparable: DimRedBase, reference: DimRedBase):
        """
        Initializes a new DimRedComparison object using comparable and reference.
        On initialization, comparable and reference are both reduced to contain only samples present in both DimRed Objects.
        In order to compare the two DimReds, on initialization Procrustes Analysis is applied transforming
        comparable towards reference. Procrustes Analysis transforms comparable by applying scaling, translation,
        rotation and reflection aiming to match all sample projections as close as possible to the projections in
        reference.

        Args:
            comparable: DimRedBase object to compare.
            reference: DimRedBase object to transform comparable towards.

        Raises
            PandoraException:
                - if either comparable of reference is not a DimRedBase object
                - if comparable and reference are not of the same type (e.g. one is PCA and the other MDS)

        """
        if not isinstance(comparable, DimRedBase) or not isinstance(reference, DimRedBase):
            raise PandoraException(
                f"comparable and reference need to be DimRedBase objects. "
                f"Instead got {type(comparable)} and {type(reference)}."
            )

        if type(comparable) != type(reference):
            raise PandoraException(
                f"comparable and reference need to be of the same DimRedBase type. "
                f"Instead got {type(comparable)} and {type(reference)}."
            )

        if isinstance(comparable, PCA):
            self.comparable, self.reference, self.disparity = match_and_transform_pcas(
                comparable=comparable, reference=reference
            )
        elif isinstance(comparable, MDS):
            match_and_transform_mds(
                comparable=comparable, reference=reference
            )
        self.sample_ids = self.comparable.embedding.sample_id

    def compare(self) -> float:
        """
        Compares self.comparable to self.reference using Procrustes Analysis and returns the similarity.

        Returns:
            float: Similarity score on a scale of 0 (entirely different) to 1 (identical) measuring the similarity of
                self.comparable and self.reference.

        """
        similarity = np.sqrt(1 - self.disparity)

        return similarity

    def compare_clustering(self, kmeans_k: int = None) -> float:
        """
        Compares the assigned cluster labels based on K-Means clustering on self.reference and self.comparable.

        Args:
            kmeans_k (int): Number k of clusters to use for K-Means clustering.
                If not set, the optimal number of clusters is determined automatically using self.reference.

        Returns:
            float: The Fowlkes-Mallow score of Cluster similarity between the clustering results
                of self.reference and self.comparable. The score ranges from 0 (entirely distinct) to 1 (identical).
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

    def _get_sample_distances(self) -> pd.Series[float]:
        """
        Computest the euclidean distances between pairs of samples in self.reference and self.comparable.

        Returns:
            pd.Series[float]: Euclidean distance between projections for each sample in
                self.reference and self.comparable. Contains one value for each sample in self.sample_ids

        """
        # make sure we are comparing the correct PC-vectors in the following
        assert np.all(
            self.comparable.embedding.sample_id == self.reference.embedding.sample_id
        )

        sample_distances = euclidean_distances(
            self.reference.embedding_matrix, self.comparable.embedding_matrix
        ).diagonal()
        return pd.Series(sample_distances, index=self.sample_ids)

    def get_sample_support_values(self) -> pd.Series[float]:
        """
        Computes the samples support value for each sample in self.sample_id using the euclidean distance
        between projections in self.reference and self.comparable.
        The euclidean distance `d` is normalized to [0, 1] by computing ` 1 / (1 + d)`.
        The higher the support the closer the projections are in euclidean space in self.reference and self.comparable.

        Returns:
            pd.Series[float]: Support value when comparing self.reference and self.comparable for each sample in self.sample_id

        """
        sample_distances = self._get_sample_distances()
        support_values = 1 / (1 + sample_distances)
        return support_values

    def detect_rogue_samples(
        self, support_value_rogue_cutoff: float = 0.5
    ) -> pd.Series[float]:
        """
        Returns the support values for all samples with a support value below support_value_rogue_cutoff.

        Args:
            support_value_rogue_cutoff (float): Threshold flagging samples as rogue. Default is 0.5.

        Returns:
            pd.Series[float]: Support values for all samples with a support value below support_value_rogue_cutoff.
                The indices of the pandas Series correspond to the sample IDs.

        """
        support_values = self.get_sample_support_values()

        rogue = support_values.loc[lambda x: (x < support_value_rogue_cutoff)]

        return rogue


def _numpy_to_dataframe(
    embedding_matrix: npt.NDArray[float],
    sample_ids: pd.Series[str],
    populations: pd.Series[str],
):
    """
    Transforms a numpy ndarray to a pandas Dataframe as required for initializing a DimRedBase object.

    Args:
        embedding_matrix (npt.NDArray[float]): Numpy ndarray containing the DimRedBase results (PC vectors) for all samples.
        sample_ids (pd.Series[str]): Pandas Series containing the sample IDs corresponding to the pc_vectors.
        populations (pd.Series[str]): Pandas Series containing the populations corresponding to the sample_ids.

    Returns:
        pd.DataFrame: Pandas dataframe containing all required columns to initialize a DimRedBase object
            (sample_id, population, D{i} for i in range(pc_vectors.shape[1]))

    """
    if embedding_matrix.ndim != 2:
        raise PandoraException(
            f"Numpy embedding matrix must be two dimensional. Passed data has {embedding_matrix.ndim} dimensions."
        )

    pca_data = pd.DataFrame(
        embedding_matrix, columns=[f"D{i}" for i in range(embedding_matrix.shape[1])]
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


def match_and_transform_pcas(comparable: PCA, reference: PCA) -> Tuple[PCA, PCA, float]:
    """
    Uses Procrustes Analysis to find a transformation matrix that most closely matches comparable to reference.
    and transforms comparable.

    Args:
        comparable (PCA): The PCA that should be transformed
        reference (PCA): The PCA that comparable should be transformed towards

    Returns:
        Tuple[PCA, PCA, float]: Two new PCA objects and the disparity. The first new PCA is the transformed comparable
            and the second one is the standardized reference. The disparity is the sum of squared distances between the
            transformed comparable and transformed reference PCAs.

    Raises:
        PandoraException:
            - Mismatch in sample IDs between comparable and reference (identical sample IDs required for comparison)
            - No samples left after clipping. This is most likely caused by incorrect annotations of sample IDs.
    """
    comparable, reference = _clip_missing_samples_for_comparison(comparable, reference)

    if not all(comparable.embedding.sample_id == reference.embedding.sample_id):
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
        reference.embedding.sample_id,
        reference.embedding.population,
    )

    standardized_reference = PCA(embedding=standardized_reference, n_components=reference.n_components,
                                 explained_variances=reference.explained_variances)

    transformed_comparable = _numpy_to_dataframe(
        transformed_comparable,
        comparable.embedding.sample_id,
        comparable.embedding.population,
    )

    transformed_comparable = PCA(embedding=transformed_comparable, n_components=comparable.n_components,
                                 explained_variances=comparable.explained_variances)

    if not all(
        standardized_reference.embedding.sample_id
        == transformed_comparable.embedding.sample_id
    ):
        raise PandoraException(
            "Sample IDS between reference and comparable don't match but is required for comparing PCA results. "
        )

    return standardized_reference, transformed_comparable, disparity


def match_and_transform_mds(comparable: MDS, reference: MDS) -> Tuple[MDS, MDS, float]:
    """
    Uses Procrustes Analysis to find a transformation matrix that most closely matches comparable to reference.
    and transforms comparable.

    Args:
        comparable (MDS): The MDS that should be transformed
        reference (MDS): The MDS that comparable should be transformed towards

    Returns:
        Tuple[MDS, MDS, float]: Two new MDS objects and the disparity. The first new MDS is the transformed comparable
            and the second one is the standardized reference. The disparity is the sum of squared distances between the
            transformed comparable and transformed reference MDSs.

    Raises:
        PandoraException:
            - Mismatch in sample IDs between comparable and reference (identical sample IDs required for comparison)
            - No samples left after clipping. This is most likely caused by incorrect annotations of sample IDs.
    """
    comparable, reference = _clip_missing_samples_for_comparison(comparable, reference)

    if not all(comparable.embedding.sample_id == reference.embedding.sample_id):
        raise PandoraException(
            "Sample IDS between reference and comparable don't match but is required for comparing MDS results. "
        )

    comp_data = comparable.embedding_matrix
    ref_data = reference.embedding_matrix

    if comp_data.shape != ref_data.shape:
        raise PandoraException(
            f"Number of samples or dimensions in comparable and reference do not match. "
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
        reference.embedding.sample_id,
        reference.embedding.population,
    )

    standardized_reference = MDS(embedding=standardized_reference, n_components=reference.n_components,
                                 stress=reference.stress)

    transformed_comparable = _numpy_to_dataframe(
        transformed_comparable,
        comparable.embedding.sample_id,
        comparable.embedding.population,
    )

    transformed_comparable = MDS(embedding=transformed_comparable, n_components=comparable.n_components,
                                 stress=comparable.stress)

    if not all(
        standardized_reference.embedding.sample_id
        == transformed_comparable.embedding.sample_id
    ):
        raise PandoraException(
            "Sample IDS between reference and comparable don't match but is required for comparing MDS results. "
        )

    return standardized_reference, transformed_comparable, disparity


