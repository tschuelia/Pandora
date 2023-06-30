from __future__ import annotations  # allows type hint PCA inside PCA class

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from pandora.custom_types import *
from pandora.custom_errors import *


class PCA:
    """Class structure for PCA results.

    This class provides methods for dealing with PCA results.
    TODO: fix docstring (e.g. types)

    Attributes:
        - pca_data (pd.DataFrame): Pandas dataframe with shape (n_samples, n_pcs + 2) that contains the PCA results.
                Has the following columns:
                    - sample_id (str or None): name for each sample, None if a np.ndarray is passed in the constructor
                    - population (str or None): population for each sample, None if a np.ndarray is passed in the constructor
                    - PC{i} for i in range(n_pcs) (float): data for the i-th PC for each sample,
                      0-indexed, so the first PC corresponds to column PC0
        - explained_variances (List[float]): List of explained variances for each PC
        - n_pcs (int): number of principal components
        - pc_vectors: TODO
    """

    def __init__(
        self,
        pca_data: Union[pd.DataFrame, npt.NDArray[float]],
        explained_variances: npt.NDArray[float],
        n_pcs: int,
        sample_ids: List[str] = None,
        populations: List[str] = None,
    ):
        """
        TODO: Documentation
        - auf sample_id Wichtigkeit hinweisen (Vergleichbarkeit!)

        """
        if explained_variances.ndim != 1:
            raise PandoraException(f"Explained variance should be a 1D numpy array. "
                                   f"Instead got {explained_variances.ndim} dimensions.")
        if explained_variances.shape[0] != n_pcs:
            raise PandoraException(
                f"Explained variance required for each PC. Got {n_pcs} but {len(explained_variances)} variances."
            )

        self.n_pcs = n_pcs
        self.explained_variances = explained_variances

        if isinstance(pca_data, np.ndarray):
            if pca_data.ndim != 2:
                raise PandoraException(
                    f"Numpy PCA data must be two dimensional. Passed data has {pca_data.ndim} dimensions."
                )

            if pca_data.shape[1] != n_pcs:
                raise PandoraException(
                    f"Numpy PCA data needs to be of shape (n_samples, n_pcs={n_pcs}). "
                    f"Instead got {pca_data.shape[1]} PCs."
                )

            pca_data = pd.DataFrame(
                pca_data, columns=[f"PC{i}" for i in range(self.n_pcs)]
            )

        if sample_ids is not None:
            if len(sample_ids) != pca_data.shape[0]:
                raise PandoraException(
                    f"One sample ID required for each sample. Got {len(sample_ids)} IDs, "
                    f"but pca_data has {pca_data.shape[0]} samples."
                )
            # overwrite/set sample IDs
            pca_data["sample_id"] = sample_ids

        if "sample_id" not in pca_data.columns:
            pca_data["sample_id"] = None

        if populations is not None:
            if len(populations) != pca_data.shape[0]:
                raise PandoraException(
                    f"One population required for each sample. Got {len(populations)} populations, "
                    
                    f"but pca_data has {pca_data.shape[0]} samples."
                )
            # overwrite/set populations
            pca_data["population"] = populations

        if "population" not in pca_data.columns:
            pca_data["population"] = None

        self.pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

        self.pc_vectors = self._get_pca_data_numpy()

    def _get_pca_data_numpy(self) -> np.ndarray:
        """
        Converts the PCA data to a numpy array.

        Returns:
             np.ndarray: Array of shape (x, y) with x = number of samples and y = self.n_pcs.
                 Does not contain the sample IDs or populations.
        """
        return self.pca_data[[f"PC{i}" for i in range(self.n_pcs)]].to_numpy()

    def get_optimal_kmeans_k(self, k_boundaries: Tuple[int, int] = None) -> int:
        """
        Determines the optimal number of clusters k for K-Means clustering according to the Bayesian Information Criterion (BIC).

        Args:
            k_boundaries (Tuple[int, int]): Minimum and maximum number of clusters. If None is given, determine the boundaries automatically.
                        If self.pca_data.populations is not identical for all samples, use the number of distinct populations,
                        otherwise use the square root of the number of samples as maximum max_k.
                        The minimum min_k is min(max_k, 3).

        Returns:
            int: the optimal number of clusters between min_n and max_n
        """
        if k_boundaries is None:
            # check whether there are distinct populations given
            n_populations = self.pca_data.population.unique().shape[0]
            if n_populations > 1:
                max_k = n_populations
            else:
                # if only one population: use the square root of the number of samples
                max_k = int(math.sqrt(self.pca_data.shape[0]))
            min_k = min(3, max_k)
        else:
            min_k, max_k = k_boundaries

        grid_search = GridSearchCV(
            estimator=GaussianMixture(),
            param_grid={"n_components": range(min_k, max_k)},
            scoring=lambda estimator, X: -estimator.bic(X),
        )

        grid_search.fit(self.pc_vectors)
        return grid_search.best_params_["n_components"]

    def cluster(self, kmeans_k: int = None) -> KMeans:
        """
        Fits a K-Means cluster to the pca data and returns a scikit-learn fitted KMeans object.

        Args:
            kmeans_k (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.

        Returns:
            KMeans: Scikit-learn KMeans object that is fitted to self.pca_data.
        """
        pca_data_np = self.pc_vectors
        if kmeans_k is None:
            kmeans_k = self.get_optimal_kmeans_k()
        kmeans = KMeans(random_state=42, n_clusters=kmeans_k, n_init=10)
        kmeans.fit(pca_data_np)
        return kmeans


def check_smartpca_results(evec: FilePath, eval: FilePath):
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

    # check the eval file: each line should cotain a single float only
    for line in eval.open():
        line = line.strip()
        try:
            float(line)
        except ValueError:
            raise PandoraException(
                f"SmartPCA eval result file appears to be incorrect: {eval}"
            )


def from_smartpca(evec: FilePath, eval: FilePath) -> PCA:
    """
    Creates a PCA object from a smartPCA results file.

    Args:
        evec (FilePath): FilePath to the evec results of SmartPCA run.
        eval (FilePath): FilePath to the eval results of SmartPCA run.

    Returns:
        PCA: PCA object encapsulating the results of the SmartPCA run.

    """
    # make sure both files are in correct format
    check_smartpca_results(evec, eval)
    # First, read the eigenvectors and transform it into the pca_data pandas dataframe
    with open(evec) as f:
        # first line does not contain data we are interested in
        f.readline()
        pca_data = pd.read_table(f, delimiter=" ", skipinitialspace=True, header=None)

    n_pcs = pca_data.shape[1] - 2

    cols = ["sample_id", *[f"PC{i}" for i in range(n_pcs)], "population"]
    pca_data = pca_data.rename(columns=dict(zip(pca_data.columns, cols)))
    pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

    # next, read the eigenvalues and compute the explained variances for all n_pcs principal components
    eigenvalues = open(eval).readlines()
    eigenvalues = [float(ev) for ev in eigenvalues]
    explained_variances = [ev / sum(eigenvalues) for ev in eigenvalues]
    # keep only the first n_pcs explained variances
    explained_variances = np.asarray(explained_variances[:n_pcs])

    return PCA(pca_data=pca_data, explained_variances=explained_variances, n_pcs=n_pcs)
