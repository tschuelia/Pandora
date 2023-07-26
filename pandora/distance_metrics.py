from sklearn.metrics.pairwise import *

from pandora.custom_errors import *
from pandora.custom_types import *


def euclidean_sample_distance(
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    return euclidean_distances(input_data, input_data), populations


def manhattan_sample_distance(
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    return manhattan_distances(input_data, input_data), populations


def population_distance(
    input_data: npt.NDArray,
    populations: pd.Series,
    distance_metric: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
) -> Tuple[npt.NDArray, pd.Series]:
    input_data = pd.DataFrame(input_data)

    if input_data.shape[0] != populations.shape[0]:
        raise PandoraException(
            f"Need to pass a population for each sample in input_data. "
            f"Got {input_data.shape[0]} samples but only {populations.shape[0]} populations."
        )

    input_data["population"] = populations

    unique_populations = populations.unique()
    n_populations = unique_populations.shape[0]
    distance_matrix = np.empty(shape=(n_populations, n_populations))

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
    return population_distance(input_data, populations, euclidean_distances)


def manhattan_population_distance(
    input_data: npt.NDArray, populations: pd.Series
) -> Tuple[npt.NDArray, pd.Series]:
    return population_distance(input_data, populations, manhattan_distances)


DISTANCE_METRICS = [
    euclidean_sample_distance,
    manhattan_sample_distance,
    euclidean_population_distance,
    manhattan_population_distance,
]
