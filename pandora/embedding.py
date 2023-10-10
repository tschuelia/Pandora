from __future__ import annotations  # allows type hint PCA inside PCA class

import math
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from pandora.custom_errors import PandoraException


class Embedding:
    """Base Wrapper class for the result of a PCA/MDS embedding."""

    def __init__(self, embedding: pd.DataFrame, n_components: int):
        if "sample_id" not in embedding.columns:
            raise PandoraException("Column `sample_id` required.")

        if "population" not in embedding.columns:
            raise PandoraException("Column `population` required.")

        if embedding.shape[1] != n_components + 2:
            # two extra columns (sample_id, population)
            raise PandoraException(
                f"One data column required for each PC. Got {n_components} but {embedding.shape[1] - 2} PC columns."
            )

        embedding_columns = [f"D{i}" for i in range(n_components)]
        if not all(c in embedding.columns for c in embedding_columns):
            raise PandoraException(
                f"Expected all of the following columns to be present in embedding: {embedding_columns}."
                f"Instead got {[c for c in embedding.columns if c not in ['sample_id', 'population']]}"
            )

        self.embedding = embedding.sort_values(by="sample_id").reset_index(drop=True)
        self.n_components = n_components
        self.embedding_matrix = self._get_embedding_numpy_array()
        self.sample_ids = self.embedding.sample_id
        self.populations = self.embedding.population

    def _get_embedding_numpy_array(self) -> np.ndarray:
        """Converts the embedding data to a numpy array.

        Returns
        -------
         np.ndarray
            Array of shape (n_samples, self.n_components). Does not contain the sample IDs or populations.
        """
        return self.embedding[[f"D{i}" for i in range(self.n_components)]].to_numpy()

    def get_optimal_kmeans_k(self, k_boundaries: Tuple[int, int] = None) -> int:
        """Determines the optimal number of clusters k for K-Means clustering according to the
        Bayesian Information Criterion (BIC).

        Parameters
        ----------
        k_boundaries : Tuple[int, int], default=None
            Minimum and maximum number of clusters. If None is given,
            determine the boundaries automatically.
            If `self.embedding.populations` is not identical for all samples, use the number of distinct populations,
            otherwise use the square root of the number of samples as maximum `max_k`.
            The minimum `min_k` is `min(max_k, 3)`.

        Returns
        -------
        int
            the optimal number of clusters between `min_n` and `max_n`

        """
        if k_boundaries is None:
            # check whether there are distinct populations given
            n_populations = self.embedding.population.unique().shape[0]
            if n_populations > 1:
                max_k = n_populations
            else:
                # if only one population: use the square root of the number of samples
                max_k = int(math.sqrt(self.embedding.shape[0]))
            min_k = min(3, max_k)
        else:
            min_k, max_k = k_boundaries

        grid_search = GridSearchCV(
            estimator=GaussianMixture(),
            param_grid={"n_components": range(min_k, max_k)},
            scoring=lambda estimator, X: -estimator.bic(X),
        )

        grid_search.fit(self.embedding_matrix)
        return grid_search.best_params_["n_components"]

    def cluster(self, kmeans_k: int = None) -> KMeans:
        """Fits a K-Means cluster to the embedding data and returns a scikit-learn fitted KMeans object.

        Parameters
        ----------
        kmeans_k : int
            Number of clusters. If not set, the optimal number of clusters is determined automatically.

        Returns
        -------
        KMeans
            Scikit-learn KMeans object that is fitted to self.embedding.

        """
        if kmeans_k is None:
            kmeans_k = self.get_optimal_kmeans_k()
        kmeans = KMeans(random_state=42, n_clusters=kmeans_k, n_init=10)
        kmeans.fit(self.embedding_matrix)
        return kmeans


class PCA(Embedding):
    """Class structure encapsulating PCA results.

    This class provides a wrapper for PCA results.

    Parameters
    ----------
    embedding : pd.DataFrame
        Pandas dataframe containing the sample ID, population and PC-Vector of all samples.
        The dataframe should contain one row per sample.
        Pandora expects the following columns:

            - sample_id (str): ID of the respective sample.
            - population (str): Name of the respective population.
            - D{i} for i in range(n_components) (float): data for the i-th PC for each sample,
              0-indexed, so the first PC corresponds to column D0
    n_components : int
        number of principal components corresponding to the PCA data
    explained_variances : npt.NDArray[float]
        Numpy ndarray containing the explained variances for each PC (shape=(n_components,))

    Attributes
    ----------
    embedding : pd.DataFrame
        Pandas dataframe with shape (n_samples, n_components + 2) that contains the PCA results.
        The dataframe contains one row per sample and has the following columns:

            - sample_id (str): ID of the respective sample.
            - population (str): Name of the respective population.
            - D{i} for i in range(n_components) (float): data for the i-th PC for each sample,
              0-indexed, so the first PC corresponds to column PC0
    explained_variances : npt.NDArray[float]
        Numpy ndarray containing the explained variances for each PC (shape=(n_components,))
    n_components : int
        number of principal components
    embedding_matrix : npt.NDArray[float]
        Numpy ndarray of shape (n_samples, n_components) containing the PCA result matrix.
    sample_ids : pd.Series[str]
        Pandas series containing the IDs of all samples.
    populations : pd.Series[str]
        Pandas series containing the population for each sample in sample_ids.

    Raises
    ------
    PandoraException
        - If `explained_variances` is not a 1D numpy array or contains more/fewer values than `n_components`.
        - If `embedding` does not contain a "sample_id" column.
        - If `embedding` does not contain a "population" column
        - If `embedding` does not contain (the correct amount of) `D{i}` columns

    """

    def __init__(
        self,
        embedding: pd.DataFrame,
        n_components: int,
        explained_variances: npt.NDArray[float],
    ):
        if explained_variances.ndim != 1:
            raise PandoraException(
                f"Explained variance should be a 1D numpy array. "
                f"Instead got {explained_variances.ndim} dimensions."
            )
        if explained_variances.shape[0] != n_components:
            raise PandoraException(
                f"Explained variance required for each PC. Got {n_components} but {len(explained_variances)} variances."
            )

        self.explained_variances = explained_variances
        super().__init__(embedding, n_components)


class MDS(Embedding):
    """Initializes a new MDS object.

    Parameters
    ----------
    embedding : pd.DataFrame
        Pandas dataframe containing the sample ID, population and embedding vector of all samples.
        The dataframe should contain one row per sample.
        Pandora expects the following columns:

            - sample_id (str): ID of the respective sample.
            - population (str): Name of the respective population.
            - D{i} for i in range(n_components) (float): data for the i-th embedding dimension for each sample,
              0-indexed, so the first dimension corresponds to column D0
    n_components : int
        number of components the data was fitted for
    stress : float
        Stress of the fitted MDS.

    Attributes
    ----------
    embedding : pd.DataFrame
        Pandas dataframe containing the sample ID, population and embedding vector of all samples.
        The dataframe should contain one row per sample.
        Pandora expects the following columns:

            - sample_id (str): ID of the respective sample.
            - population (str): Name of the respective population.
            - D{i} for i in range(n_components) (float): data for the i-th embedding dimension for each sample,
              0-indexed, so the first dimension corresponds to column D0
    n_components : int
        number of components the data was fitted for
    stress : float
        Stress of the fitted MDS.
    embedding_matrix : npt.NDArray[float]
        Numpy ndarray of shape (n_samples, n_components) containing the MDS result matrix.
    sample_ids : pd.Series[str]
        Pandas series containing the IDs of all samples.
    populations : pd.Series[str]
        Pandas series containing the population for each sample in sample_ids.

    Raises
    ------
    PandoraException
        - If `embedding` does not contain a "sample_id" column.
        - If `embedding` does not contain a "population" column.
        - If `embedding` does not contain (the correct amount of) `D{i}` columns.

    """

    def __init__(self, embedding: pd.DataFrame, n_components: int, stress: float):
        self.stress = stress
        super().__init__(embedding, n_components)


def check_smartpca_results(evec: pathlib.Path, eval: pathlib.Path) -> None:
    """Checks whether the smartpca results finished properly and contain all required information.

    Parameters
    ----------
    evec : pathlib.Path
        Filepath pointing to a .evec result file of a smartpca run.
    eval : pathlib.Path
        Filepath pointing to a .eval result file of a smartpca run.

    Returns
    -------
    None

    Raises
    ------
    PandoraException
        If either the evec file or the eval file are incorrect.

    """
    # check the evec file:
    # - first line should start with #eigvals: and then determines the number of PCs
    with evec.open() as f:
        line = f.readline().strip()
        if not line.startswith("#eigvals"):
            raise PandoraException(
                f"SmartPCA evec result file appears to be incorrect: {evec}"
            )

        variances = line.split()[1:]
        try:
            [float(v) for v in variances]
        except ValueError:
            raise PandoraException(
                f"SmartPCA evec result file appears to be incorrect: {evec}"
            )
        n_pcs = len(variances)

        # all following lines should look like this:
        # SampleID  PC0  PC1  ...  PCN-1  Population
        for line in f.readlines():
            values = line.strip().split()
            if len(values) != n_pcs + 2:
                raise PandoraException(
                    f"SmartPCA evec result file appears to be incorrect: {evec}"
                )

            # all PC values should be floats
            try:
                [float(v) for v in values[1:-1]]
            except ValueError:
                raise PandoraException(
                    f"SmartPCA evec result file appears to be incorrect: {evec}"
                )

    # check the eval file: each line should contain a single float only
    for line in eval.open():
        line = line.strip()
        try:
            float(line)
        except ValueError:
            raise PandoraException(
                f"SmartPCA eval result file appears to be incorrect: {eval}"
            )


def from_smartpca(evec: pathlib.Path, eval: pathlib.Path) -> PCA:
    """Creates a PCA object based on the results of a smartpca run

    Parameters
    ----------
    evec : pathlib.Path
        Filepath pointing to a .evec result file of a smartpca run.
    eval : pathlib.Path
        Filepath pointing to a .eval result file of a smartpca run.

    Returns
    -------
    PCA
        PCA object of the results of the respective smartpca run.

    Raises
    ------
    PandoraException
        If either the evec file or the eval file are incorrect.

    """
    # make sure both files are in correct file_format
    check_smartpca_results(evec, eval)
    # First, read the eigenvectors and transform it into the pca_data pandas dataframe
    with open(evec) as f:
        # first line does not contain data we are interested in
        f.readline()
        pca_data = pd.read_table(f, delimiter=" ", skipinitialspace=True, header=None)

    n_pcs = pca_data.shape[1] - 2

    cols = ["sample_id", *[f"D{i}" for i in range(n_pcs)], "population"]
    pca_data = pca_data.rename(columns=dict(zip(pca_data.columns, cols)))
    pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

    # next, read the eigenvalues and compute the explained variances for all n_components principal components
    eigenvalues = open(eval).readlines()
    eigenvalues = [float(ev) for ev in eigenvalues]
    explained_variances = [ev / sum(eigenvalues) for ev in eigenvalues]
    # keep only the first n_components explained variances
    explained_variances = np.asarray(explained_variances[:n_pcs])

    return PCA(
        embedding=pca_data, n_components=n_pcs, explained_variances=explained_variances
    )


def from_sklearn_mds(
    embedding: pd.DataFrame,
    sample_ids: pd.Series,
    populations: pd.Series,
    stress: float,
) -> MDS:
    """Creates a new MDS object based on an MDS embedding pandas dataframe.

    Note that embedding is expected to have a
    column entitled populations. This is needed since the input distance matrices for MDS may be summary statistics
    for all samples of one population. The resulting MDS object however will duplicate the results for each sample
    given in sample_ids to match the original input data.

    Parameters
    ----------
    embedding : pd.DataFrame
        MDS embedding data as pandas DataFrame. Each row corresponds to a single sample or population.
        The embedding is expected to  have a column entitled 'population' denoting the respective population of the row.
    sample_ids : pd.Series
        Pandas Series containing IDs of samples the embedding data is for. Note that the number of sample IDs can be
        larger than the number of rows in the embedding. This is the case if the embedding was computed per population
        but the data should be mapped for each sample. The number of sample IDs needs to match the number of populations.
    populations : pd.Series
        Pandas Series containing the population for each sample in sample_ids. The number of populations needs to match
        the number of sample IDs.
    stress : float
        Goodness of the MDS fit for the data.

    Returns
    -------
    MDS
        MDS object encapsulating the MDS data

    Raises
    ------
    PandoraException
        - If `embedding` does not contain a "populations" column.
        - If the number of samples and number of populations are not identical. Exactly population is required for each sample.

    """
    if "population" not in embedding.columns:
        raise PandoraException(
            "The embedding dataframe needs to contain a column entitled 'populations'."
        )

    n_components = embedding.shape[1] - 1  # one column is 'populations'

    if sample_ids.shape[0] != populations.shape[0]:
        raise PandoraException(
            "Number of sample IDs needs to be identical to the number of populations."
            f"Got {sample_ids.shape[0]} sample IDs but {populations.shape[0]} populations."
        )

    # depending on whether the distance matrix for MDS was computed using all samples or per population
    # we need to duplicate results for all samples per population
    # so we first check if the embedding shape matches the number of populations
    if embedding.shape[0] == populations.shape[0]:
        # we can directly use the embedding data as mds_data
        mds_data = embedding
        # add the sample_id column
        mds_data["sample_id"] = sample_ids
    else:
        # otherwise we need to iterate the embedding data and duplicate the results for each sample in sample_ids
        mds_data = []

        for sample_id, population in zip(sample_ids, populations):
            embedding_vector = embedding.loc[
                lambda x: x.population.str.strip() == population
            ]

            assert embedding_vector.shape[0] == 1, (
                f"Multiple/No MDS embeddings for population {population}. "
                f"Got {embedding_vector.shape[0]} rows but expected exactly 1."
            )
            embedding_vector = embedding_vector.squeeze().to_list()
            mds_data.append([sample_id, *embedding_vector])

        mds_data = pd.DataFrame(
            data=mds_data,
            columns=[
                "sample_id",
                *[f"D{i}" for i in range(n_components)],
                "population",
            ],
        )
    return MDS(mds_data, n_components, stress)
