import itertools
from typing import Callable, Optional, Tuple

import allel
import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from pandora.custom_errors import PandoraException
from pandora.imputation import impute_data


def _check_input(input_data: npt.NDArray, populations: pd.Series):
    if input_data.shape[0] != populations.shape[0]:
        raise PandoraException(
            f"Need to pass a population for each sample in input_data. "
            f"Got {input_data.shape[0]} samples but only {populations.shape[0]} populations."
        )


def euclidean_sample_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise Euclidean distances between all samples (rows) in
    ``input_data``.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``. Note that this population info is not
        used for distance computation as the distance is computed per sample. The parameter is only required to provide
        a unique interface for per-sample and per-population distances.
    imputation : Optional[str]
        Imputation method to use. Available options are:\n
        - ``"mean"``: Imputes missing values with the average of the respective SNP
        - ``"remove"``: Removes all SNPs with at least one missing value.
        - ``None``: Note that this option is only valid if input_data does not contain NaN values.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise Euclidean distances between all samples.
        The array is of shape ``(n_samples, n_samples)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This is identical to the passed
        series of populations since this information is not used for distance computation.

    Raises
    ------
    PandoraException
        If imputation is ``None`` but ``input_data`` contains NaN values.
    """
    if imputation is None and np.isnan(input_data).any():
        raise PandoraException(
            "Imputation method cannot be None if input_data contains NaN values."
        )

    input_data = impute_data(input_data, imputation)
    _check_input(input_data, populations)
    return euclidean_distances(input_data, input_data), populations


def manhattan_sample_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise manhattan distances between all samples (rows) in
    input_data.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``. Note that this population info is not
        used for distance computation as the distance is computed per sample. The parameter is only required to provide
        a unique interface for per-sample and per-population distances.
    imputation : Optional[str]
        Imputation method to use. Available options are:\n
        - ``"mean"``: Imputes missing values with the average of the respective SNP
        - ``"remove"``: Removes all SNPs with at least one missing value.
        - None: Note that this option is only valid if input_data does not contain NaN values.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise manhattan distances between all samples.
        The array is of shape ``(n_samples, n_samples)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This is identical to the passed
        series of populations since this information is not used for distance computation.

    Raises
    ------
    PandoraException
        If imputation is ``None`` but ``input_data`` contains NaN values.
    """
    if imputation is None and np.isnan(input_data).any():
        raise PandoraException(
            "Imputation method cannot be None if input_data contains NaN values."
        )

    input_data = impute_data(input_data, imputation)
    _check_input(input_data, populations)
    return manhattan_distances(input_data, input_data), populations


def hamming_sample_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise hamming distances between all samples (rows) in input_data.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``. Note that this population info is not
        used for distance computation as the distance is computed per sample. The parameter is only required to provide
        a unique interface for per-sample and per-population distances.
    imputation : Optional[str]
        Imputation method to use. This parameter is not used and exists only for compatibility with the interface
        required for the ``dataset.run_mds`` method.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise hamming distances between all samples.
        The array is of shape ``(n_samples, n_samples)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This is identical to the passed
        series of populations since this information is not used for distance computation.
    """
    n_samples = input_data.shape[0]
    distance_matrix = np.zeros(shape=(n_samples, n_samples))

    for (i, s1), (j, s2) in itertools.combinations(enumerate(input_data), r=2):
        hamming_distance = np.nansum([abs(v1 - v2) for v1, v2 in zip(s1, s2)])
        distance_matrix[i, j] = hamming_distance
        # distance matrix should be symmetric
        distance_matrix[j, i] = hamming_distance

    return distance_matrix, populations


def missing_corrected_hamming_sample_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise, hamming distances between all samples (rows) in input_data.
    Compared to ``hamming_sample_distance``, this method additionally corrects for missing samples (see Notes below).

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``. Note that this population info is not
        used for distance computation as the distance is computed per sample. The parameter is only required to provide
        a unique interface for per-sample and per-population distances.
    imputation : Optional[str]
        Imputation method to use. This parameter is not used and exists only for compatibility with the interface
        required for the ``dataset.run_mds`` method.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise hamming distances between all samples.
        The array is of shape ``(n_samples, n_samples)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This is identical to the passed
        series of populations since this information is not used for distance computation.

    Notes
    -----
    Instead of the plain hamming distance :math:`d(i, j)` between two samples :math:`i` and :math:`j`, it corrects for
    the fraction of missing data in both samples (:math:`m_i`, :math:`m_j`). However, for the correction, we only
    consider missing values if they are missing in either of the two samples, but not in both. We denote the fraction of
    data missing in both samples as :math:`m_{i,j}`
    Thus, the missing correct hamming distance :math:`d_m(i, j)` computes as:

    .. math:: d_m(i, j) = \\frac{d(i, j)}{m_i + m_j - m_{i, j}}

    Note that this distance metric corresponds to the ``PLINK --distance 'flat-missing'`` computation.
    """
    n_samples = input_data.shape[0]
    distance_matrix = np.zeros(shape=(n_samples, n_samples))

    for (i, s1), (j, s2) in itertools.combinations(enumerate(input_data), r=2):
        hamming_distance = np.nansum([abs(v1 - v2) for v1, v2 in zip(s1, s2)])

        # compute the fraction of missing values in both sequences
        missing_s1 = np.sum(np.isnan(s1)) / len(s1)
        missing_s2 = np.sum(np.isnan(s2)) / len(s2)
        # compute the fraction of missing values
        missing_in_both = np.sum(
            [np.isnan(v1) and np.isnan(v2) for v1, v2 in zip(s1, s2)]
        ) / len(s1)

        normalization_factor = missing_s1 + missing_s2 - missing_in_both
        hamming_distance /= 1 - normalization_factor
        distance_matrix[i, j] = hamming_distance
        # distance matrix should be symmetric
        distance_matrix[j, i] = hamming_distance

    return distance_matrix, populations


def population_distance(
    input_data: npt.NDArray,
    populations: pd.Series,
    distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise distances between all unique populations using the provided
    distance metric.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in input_data.
    distance_metric : Callable[[npt.NDArray, npt.NDArray, str], npt.NDArray]
        Distance metric function to use for the pairwise population distance computation. Needs to be a callable that
        takes two numpy arrays as input (each array contains the data of all samples in ``input_data`` for one specific
        population) and returns the pairwise distance between unique pairs of samples of both populations.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise distances between all unique populations.
        The array is of shape ``(n_unique_populations, n_unique_populations)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.
    """
    _check_input(input_data, populations)
    input_data = pd.DataFrame(input_data)

    input_data["population"] = populations

    unique_populations = populations.unique()
    n_populations = unique_populations.shape[0]
    distance_matrix = np.zeros(shape=(n_populations, n_populations))

    # now for each combination of populations, compute the average sample distance
    for (i, p1), (j, p2) in itertools.combinations(enumerate(unique_populations), r=2):
        samples_p1 = input_data.loc[lambda x: x.population == p1]
        samples_p2 = input_data.loc[lambda x: x.population == p2]

        # we only need the geno type data, so remove the population column
        samples_p1 = samples_p1.loc[:, samples_p1.columns != "population"]
        samples_p2 = samples_p2.loc[:, samples_p2.columns != "population"]

        # compute the pairwise sample distances for all samples in p1 and p2
        sample_distances = distance_metric(samples_p1, samples_p2)
        # average over all pairwise comparisons to get the population distance for p1 and p2
        distance_matrix[i, j] = np.mean(sample_distances)
        # distance matrix needs to be symmetric
        distance_matrix[j, i] = np.mean(sample_distances)

    return distance_matrix, pd.Series(unique_populations)


def euclidean_population_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise Euclidean distances between all unique populations.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``.
    imputation : Optional[str]
        Imputation method to use. Available options are:\n
        - ``"mean"``: Imputes missing values with the average of the respective SNP
        - ``"remove"``: Removes all SNPs with at least one missing value.
        - ``None``: Note that this option is only valid if ``input_data`` does not contain NaN values.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise Euclidean distances between all unique populations.
        The array is of shape ``(n_unique_populations, n_unique_populations)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.

    Raises
    ------
    PandoraException
        If imputation is ``None`` but ``input_data`` contains NaN values.
    """
    if imputation is None and np.isnan(input_data).any():
        raise PandoraException(
            "Imputation method cannot be None if input_data contains NaN values."
        )

    input_data = impute_data(input_data, imputation)
    _check_input(input_data, populations)
    return population_distance(input_data, populations, euclidean_distances)


def manhattan_population_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise manhattan distances between all unique populations.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in ``input_data``.
    imputation : Optional[str]
        Imputation method to use. Available options are:\n
        - ``"mean"``: Imputes missing values with the average of the respective SNP
        - ``"remove"``: Removes all SNPs with at least one missing value.
        - ``None``: Note that this option is only valid if ``input_data`` does not contain NaN values.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise manhattan distances between all unique populations.
        The array is of shape ``(n_unique_populations, n_unique_populations)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.

    Raises
    ------
    PandoraException
        If imputation is ``None`` but ``input_data`` contains NaN values.
    """
    if imputation is None and np.isnan(input_data).any():
        raise PandoraException(
            "Imputation method cannot be None if input_data contains NaN values."
        )

    input_data = impute_data(input_data, imputation)
    _check_input(input_data, populations)
    return population_distance(input_data, populations, manhattan_distances)


def fst_population_distance(
    input_data: npt.NDArray, populations: pd.Series, imputation: Optional[str]
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise FST distances between all unique populations.

    Parameters
    ----------
    input_data : npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations : pd.Series[str]
        Pandas Series containing a population name for each row in input_data.
    imputation : Optional[str]
        Imputation method to use. For the FST populations distance, only ``imputation=None`` is supported.

    Returns
    -------
    distance_matrix : npt.NDArray
        Distance matrix of pairwise FST distances between all unique populations.
        The array is of shape ``(n_unique_populations, n_unique_populations)``.
    populations : pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.
    """
    if imputation is not None:
        raise PandoraException(
            "Currently the FST distance metric only supports imputation=None."
        )

    _check_input(input_data, populations)
    # 1. transform input_data into scikit-allel GenotypeArray
    geno_array = _geno_data_to_geno_array(input_data)
    n_snps = geno_array.n_variants

    # 2. get the list of subpopulations
    # to do so, select the indices of all unique populations
    subpopulations = []
    for population in populations.unique():
        population_indices = populations.index[populations == population].tolist()
        subpopulations.append(population_indices)

    # 3. for each pair of populations, get the fst value and store it in the fst-matrix
    n_populations = populations.unique().shape[0]
    fst_matrix = np.zeros(shape=(n_populations, n_populations))

    allel_counts = [
        geno_array.count_alleles(subpop=subpop) for subpop in subpopulations
    ]

    assert len(subpopulations) == len(allel_counts) == n_populations

    for (i, ct1), (j, ct2) in itertools.combinations(enumerate(allel_counts), r=2):
        if i == j:
            continue
        fst, *_ = allel.average_patterson_fst(ct1, ct2, n_snps)
        fst_matrix[i, j] = fst_matrix[j, i] = fst

    return fst_matrix, pd.Series(populations.unique())


def _geno_to_var(geno):
    if geno == 0:
        return [0, 0]
    elif geno == 1:
        return [0, 1]
    elif geno == 2:
        return [1, 1]
    elif np.isnan(geno):
        return [-1, -1]
    else:
        raise PandoraException(
            f"Unrecognized geno type value: {geno}. Only values 0, 1, 2 or np.nan (missing) allowed."
        )


def _geno_data_to_geno_array(geno_data):
    n_samples, n_snps = geno_data.shape

    geno_array = []
    for snp in range(n_snps):
        snp_data = [_geno_to_var(geno) for geno in geno_data[:, snp]]
        geno_array.append(snp_data)

    geno_array = np.asarray(geno_array)
    return allel.GenotypeArray(geno_array)


DISTANCE_METRICS = [
    euclidean_sample_distance,
    manhattan_sample_distance,
    hamming_sample_distance,
    missing_corrected_hamming_sample_distance,
    euclidean_population_distance,
    manhattan_population_distance,
    fst_population_distance,
]
