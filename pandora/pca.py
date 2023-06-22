from __future__ import annotations  # allows type hint PCA inside PCA class

import math
import warnings

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from pandora.custom_types import *
from pandora.plotting import get_distinct_colors


class PCA:
    """Class structure for PCA results.

    This class provides methods for dealing with PCA results.

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
        pca_data: Union[pd.DataFrame, np.ndarray],
        explained_variances: List[float],
        n_pcs: int,
        sample_ids: List[str] = None,
        populations: List[str] = None,
    ):
        """
        TODO: Documentation
        - auf sample_id Wichtigkeit hinweisen (Vergleichbarkeit!)

        """
        self.n_pcs = n_pcs
        self.explained_variances = explained_variances

        if isinstance(pca_data, np.ndarray):
            pca_data = pd.DataFrame(
                pca_data, columns=[f"PC{i}" for i in range(self.n_pcs)]
            )

        if sample_ids is not None:
            pca_data["sample_id"] = sample_ids

        if populations is not None:
            pca_data["population"] = populations

        if "sample_id" not in pca_data.columns:
            pca_data["sample_id"] = None

        if "population" not in pca_data.columns:
            pca_data["population"] = None

        self.pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

        self.pc_vectors = self._get_pca_data_numpy()

    def set_populations(self, populations: Union[List, pd.Series]):
        """
        Attributes the given populations to the PCA data. The number of populations given must be identical to the
        number of samples in the PCA data.

        Args:
             populations (Union[List, pd.Series]): A population for each sample in the PCA data.

        Raises:
            ValueError: If the number of populations does not match the number of samples in the PCA data.
        """
        if len(populations) != self.pca_data.shape[0]:
            raise ValueError(
                f"Provide a population for each sample. Got {self.pca_data.shape[0]} samples but {len(populations)} populations."
            )
        self.pca_data["population"] = populations

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

    def plot_clusters(
        self,
        pcx: int = 0,
        pcy: int = 1,
        kmeans_k: int = None,
        fig: go.Figure = None,
        **kwargs,
    ) -> go.Figure:
        """
        TODO: Docstring
        """
        show_variance_in_axes = fig is None
        fig = go.Figure() if fig is None else fig

        if kmeans_k is None:
            kmeans_k = self.get_optimal_kmeans_k()

        cluster_labels = self.cluster(kmeans_k=kmeans_k).labels_

        _pca_data = self.pca_data.copy()
        _pca_data["cluster"] = cluster_labels

        colors = get_distinct_colors(kmeans_k)

        for i in range(kmeans_k):
            _data = _pca_data.loc[_pca_data.cluster == i]
            fig.add_trace(
                go.Scatter(
                    x=_data[f"PC{pcx}"],
                    y=_data[f"PC{pcy}"],
                    mode="markers",
                    marker_color=colors[i],
                    name=f"Cluster {i + 1}",
                    **kwargs,
                )
            )

        return self._update_fig(fig, pcx, pcy, show_variance_in_axes)

    def plot_populations(
        self, pcx: int = 0, pcy: int = 1, fig: go.Figure = None, **kwargs
    ) -> go.Figure:
        """
        TODO: Docstring
        """
        show_variance_in_axes = fig is None
        fig = go.Figure() if fig is None else fig

        if self.pca_data.population.isna().all():
            raise ValueError(
                "Cannot plot populations: no populations associated with PCA data."
            )

        populations = self.pca_data.population.unique()
        colors = get_distinct_colors(len(populations))

        for i, population in enumerate(populations):
            _data = self.pca_data.loc[self.pca_data.population == population]
            fig.add_trace(
                go.Scatter(
                    x=_data[f"PC{pcx}"],
                    y=_data[f"PC{pcy}"],
                    mode="markers",
                    marker_color=colors[i],
                    name=population,
                    **kwargs,
                )
            )

        return self._update_fig(fig, pcx, pcy, show_variance_in_axes)

    def plot_projections(
        self,
        pca_populations: List[str],
        pcx: int = 0,
        pcy: int = 1,
        fig: go.Figure = None,
        **kwargs,
    ):
        """
        TODO: Docstring
        """
        show_variance_in_axes = fig is None
        fig = go.Figure() if fig is None else fig

        if len(pca_populations) == 0:
            raise ValueError(
                "It appears that all populations were used for the PCA. "
                "To plot projections provide a non-empty list of populations with which the PCA was performed!"
            )

        populations = self.pca_data.population.unique()
        projection_colors = get_distinct_colors(populations.shape[0])

        for i, population in enumerate(populations):
            _data = self.pca_data.loc[self.pca_data.population == population]
            marker_color = (
                projection_colors[i]
                if population not in pca_populations
                else "lightgray"
            )
            fig.add_trace(
                go.Scatter(
                    x=_data[f"PC{pcx}"],
                    y=_data[f"PC{pcy}"],
                    mode="markers",
                    marker_color=marker_color,
                    name=population,
                    **kwargs,
                )
            )
        return self._update_fig(fig, pcx, pcy, show_variance_in_axes)

    def _update_fig(
        self, fig: go.Figure, pcx: int, pcy: int, show_variance_in_axes: bool
    ):
        xtitle = f"PC {pcx + 1}"
        ytitle = f"PC {pcy + 1}"

        if show_variance_in_axes:
            xtitle += f" ({round(self.explained_variances[pcx] * 100, 1)}%)"
            ytitle += f" ({round(self.explained_variances[pcy] * 100, 1)}%)"

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle)
        fig.update_layout(template="plotly_white", height=1000, width=1000)

        return fig


def check_smartpca_results(evec: FilePath, eval: FilePath):
    # check the evec file:
    # - first line should start with #eigvals: and then determines the number of PCs
    with evec.open() as f:
        line = f.readline().strip()
        if not line.startswith("#eigvals"):
            raise RuntimeError(f"SmartPCA evec result file appears to be incorrect: {evec}")

        variances = line.split()[1:]
        try:
            [float(v) for v in variances]
        except ValueError:
            raise RuntimeError(f"SmartPCA evec result file appears to be incorrect: {evec}")
        n_pcs = len(variances)

        # all following lines should look like this:
        # SampleID  PC0  PC1  ...  PCN-1  Population
        for line in f.readlines():
            values = line.strip().split()
            if len(values) != n_pcs + 2:
                raise RuntimeError(f"SmartPCA evec result file appears to be incorrect: {evec}")

            # all PC values should be floats
            try:
                [float(v) for v in values[1:-1]]
            except ValueError:
                raise RuntimeError(f"SmartPCA evec result file appears to be incorrect: {evec}")

    # check the eval file: each line should cotain a single float only
    for line in eval.open():
        line = line.strip()
        try:
            float(line)
        except ValueError:
            raise RuntimeError(f"SmartPCA eval result file appears to be incorrect: {eval}")


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
    explained_variances = explained_variances[:n_pcs]

    return PCA(pca_data=pca_data, explained_variances=explained_variances, n_pcs=n_pcs)
