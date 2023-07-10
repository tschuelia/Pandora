from __future__ import annotations  # allows type hint PCA inside PCA class

import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from pandora.custom_types import *
from pandora.custom_errors import *


class Embedding:
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

    def _get_embedding_numpy_array(self) -> np.ndarray:
        """
        Converts the embedding data to a numpy array.

        Returns:
             np.ndarray: Array of shape (n_samples, self.n_components).
                 Does not contain the sample IDs or populations.
        """
        return self.embedding[[f"D{i}" for i in range(self.n_components)]].to_numpy()

    def get_optimal_kmeans_k(self, k_boundaries: Tuple[int, int] = None) -> int:
        """
        Determines the optimal number of clusters k for K-Means clustering according to the Bayesian Information Criterion (BIC).

        Args:
            k_boundaries (Tuple[int, int]): Minimum and maximum number of clusters. If None is given,
                determine the boundaries automatically.
                If self.embedding.populations is not identical for all samples, use the number of distinct populations,
                otherwise use the square root of the number of samples as maximum max_k.
                The minimum min_k is min(max_k, 3).

        Returns:
            int: the optimal number of clusters between min_n and max_n
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
        """
        Fits a K-Means cluster to the embedding data and returns a scikit-learn fitted KMeans object.

        Args:
            kmeans_k (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.

        Returns:
            KMeans: Scikit-learn KMeans object that is fitted to self.embedding.
        """
        if kmeans_k is None:
            kmeans_k = self.get_optimal_kmeans_k()
        kmeans = KMeans(random_state=42, n_clusters=kmeans_k, n_init=10)
        kmeans.fit(self.embedding_matrix)
        return kmeans


class PCA(Embedding):
    """Class structure encapsulating PCA results.

    This class provides a wrapper for PCA results.

    Attributes:
        embedding (pd.DataFrame): Pandas dataframe with shape (n_samples, n_pcs + 2) that contains the PCA results.
            The dataframe contains one row per sample and has the following columns:
                - sample_id (str): ID of the respective sample.
                - population (str): Name of the respective population.
                - D{i} for i in range(n_pcs) (float): data for the i-th PC for each sample,
                  0-indexed, so the first PC corresponds to column PC0
        explained_variances (npt.NDArray[float]): Numpy ndarray containing the explained variances for each PC (shape=(n_pcs,))
        n_pcs (int): number of principal components
        embedding_matrix: Numpy ndarray of shape (n_samples, n_pcs) containing the PCA result matrix.
    """

    def __init__(self, embedding: pd.DataFrame, n_components: int, explained_variances: npt.NDArray[float]):
        """
        Initializes a new PCA object.

        Args:
            embedding (pd.DataFrame): Pandas dataframe containing the sample ID, population and PC-Vector of all samples.
                The dataframe should contain one row per sample.
                Pandora expects the following columns:
                    - sample_id (str): ID of the respective sample.
                    - population (str): Name of the respective population.
                    - D{i} for i in range(n_pcs) (float): data for the i-th PC for each sample,
                      0-indexed, so the first PC corresponds to column PC0
            n_components (int): number of principal components
            explained_variances (npt.NDArray[float]): Numpy ndarray containing the explained variances for each PC (shape=(n_pcs,))

        Raises:
            PandoraException:
                - explained_variances is not a 1D numpy array or contains more/fewer values than n_pcs
                - embedding does not contain a "sample_id" column
                - embedding does not contain a "population" column
                - embedding does not contain (the correct amount of) D{i} columns
        """
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
    def __init__(self, embedding: pd.DataFrame, n_components: int, stress: float):
        self.stress = stress
        super().__init__(embedding, n_components)


def check_smartpca_results(evec: pathlib.Path, eval: pathlib.Path):
    """
    Checks whether the smartpca results finished properly and contain all required information.

    Args:
        evec (pathlib.Path): Filepath pointing to a .evec result file of a smartpca run.
        eval (pathlib.Path): Filepath pointing to a .eval result file of a smartpca run.

    Returns: None

    Raises:
        PandoraException: If either the evec file or the eval file are incorrect.

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
    """
    Creates a PCA object based on the results of a smartpca run

    Args:
        evec (pathlib.Path): Filepath pointing to a .evec result file of a smartpca run.
        eval (pathlib.Path): Filepath pointing to a .eval result file of a smartpca run.

    Returns:
        PCA: PCA object of the results of the respective smartpca run.

    Raises:
        PandoraException: If either the evec file or the eval file are incorrect.

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

    # next, read the eigenvalues and compute the explained variances for all n_pcs principal components
    eigenvalues = open(eval).readlines()
    eigenvalues = [float(ev) for ev in eigenvalues]
    explained_variances = [ev / sum(eigenvalues) for ev in eigenvalues]
    # keep only the first n_pcs explained variances
    explained_variances = np.asarray(explained_variances[:n_pcs])

    return PCA(embedding=pca_data, n_components=n_pcs, explained_variances=explained_variances)
