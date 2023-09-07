import itertools
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from pandora.custom_errors import PandoraException


def euclidean_sample_distance(
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise Euclidean distances between all samples (rows) in input_data.

    Parameters
    ----------
    input_data: npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations: pd.Series[str]
        Pandas Series containing a population name for each row in input_data. Note that this population info is not
        used for distance computation as the distance is computed per sample. The parameter is only required to provide
        a unique interface for per-sample and per-population distances.

    Returns
    -------
    npt.NDArray
        Distance matrix of pairwise Euclidean distances between all unique populations.
        The array is of shape (n_samples, n_samples).
    pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This is identical to the passed
        series of populations since this information is not used for distance computation.
    """
    return euclidean_distances(input_data, input_data), populations


def manhattan_sample_distance(
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise manhattan distances between all samples (rows) in input_data.

    Parameters
    ----------
    input_data: npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations: pd.Series[str]
        Pandas Series containing a population name for each row in input_data. Note that this population info is not
        used for distance computation as the distance is computed per sample. The parameter is only required to provide
        a unique interface for per-sample and per-population distances.

    Returns
    -------
    npt.NDArray
        Distance matrix of pairwise manhattan distances between all unique populations.
        The array is of shape (n_samples, n_samples).
    pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This is identical to the passed
        series of populations since this information is not used for distance computation.
    """
    return manhattan_distances(input_data, input_data), populations


def population_distance(
    input_data: npt.NDArray,
    populations: pd.Series,
    distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise distances between all unique populations using the provided distance metric.

    Parameters
    ----------
    input_data: npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations: pd.Series[str]
        Pandas Series containing a population name for each row in input_data.
    distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
        Distance metric function to use for pairwise distance computation. The distance metric needs to be a callable
        that takes a numpy array (input_data) and the respective population for each row in the input_data as input and
        returns the pairwise distance between all unique populations as numpy matrix.

    Returns
    -------
    npt.NDArray
        Distance matrix of pairwise distances between all unique populations.
        The array is of shape (n_unique_populations, n_unique_populations).
    pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.

    """
    input_data = pd.DataFrame(input_data)

    if input_data.shape[0] != populations.shape[0]:
        raise PandoraException(
            f"Need to pass a population for each sample in input_data. "
            f"Got {input_data.shape[0]} samples but only {populations.shape[0]} populations."
        )

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
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise Euclidean distances between all unique populations.

    Parameters
    ----------
    input_data: npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations: pd.Series[str]
        Pandas Series containing a population name for each row in input_data.

    Returns
    -------
    npt.NDArray
        Distance matrix of pairwise Euclidean distances between all unique populations.
        The array is of shape (n_unique_populations, n_unique_populations).
    pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.
    """
    return population_distance(input_data, populations, euclidean_distances)


def manhattan_population_distance(
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    """Computes and returns the distance matrix of pairwise manhattan distances between all unique populations.

    Parameters
    ----------
    input_data: npt.NDArray
        Numpy Array containing the genetic input data to use.
    populations: pd.Series[str]
        Pandas Series containing a population name for each row in input_data.

    Returns
    -------
    npt.NDArray
        Distance matrix of pairwise manhattan distances between all unique populations.
        The array is of shape (n_unique_populations, n_unique_populations).
    pd.Series
        Pandas Series containing a population name for each row in the distance matrix. This values of this series are
        the unique populations.
    """
    return population_distance(input_data, populations, manhattan_distances)


DISTANCE_METRICS = [
    euclidean_sample_distance,
    manhattan_sample_distance,
    euclidean_population_distance,
    manhattan_population_distance,
]
