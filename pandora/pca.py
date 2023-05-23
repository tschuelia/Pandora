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
from pandora.utils import get_colors


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
            scoring=lambda estimator, X: -estimator.bic(X)
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

    def _plot_clusters(
        self,
        pcx: int = 0,
        pcy: int = 1,
        kmeans_k: int = None,
        fig: go.Figure = None,
        **kwargs,
    ) -> go.Figure:
        kmeans_k = (
            kmeans_k if kmeans_k is not None else self.get_optimal_kmeans_k()
        )
        cluster_labels = self.cluster(kmeans_k=kmeans_k).labels_

        _pca_data = self.pca_data.copy()
        _pca_data["cluster"] = cluster_labels

        colors = get_colors(kmeans_k)

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

        return fig

    def _plot_populations(
        self, pcx: int = 0, pcy: int = 1, fig: go.Figure = None, **kwargs
    ) -> go.Figure:
        if self.pca_data.population.isna().all():
            raise ValueError(
                "Cannot plot populations: no populations associated with PCA data."
            )
        populations = self.pca_data.population.unique()
        colors = get_colors(len(populations))
        assert len(populations) == len(colors), f"{len(populations)}, {len(colors)}"
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

        return fig

    def plot(
        self,
        pcx: int = 0,
        pcy: int = 1,
        annotation: str = None,
        kmeans_k: int = None,
        fig: go.Figure = None,
        marker_color="darkseagreen",
        name: str = "",
        outfile: FilePath = None,
        redo: bool = False,
        **kwargs,
    ) -> go.Figure:
        """
        Plots the PCA data for pcx and pcy.
        TODO: update args description

        Args:
            pcx (int): Number of the PC to plot on the x-axis.
                The PCs are 0-indexed, so to plot the first principal component, set pcx = 0. Defaults to 0.
            pcy (int): Number of the PC to plot on the y-axis.
                The PCs are 0-indexed, so to plot the second principal component, set pcy = 1. Defaults to 1.
            annotation (bool): If None, plots alls samples with the same color.
                If "population", plots each population with a different color.
                If "cluster", applies K-Means clustering and plots each cluster with a different color.
            kmeans_k (int): TODO
            fig (go.Figure): If set, appends the PCA data to this fig. Default is to plot on a new, empty figure.
            marker_color (str): TODO
            name (str): Name of the trace in the resulting plot. Setting a name will only have an effect if
                plot_populations = False and fig is not None.
            outfile (FilePath): TODO
            redo (bool): TODO

        Returns:
            go.Figure: Plotly figure containing a scatter plot of the PCA data.
        """
        show_variance_in_axes = True
        if not fig:
            fig = go.Figure()
            show_variance_in_axes = False

        if kmeans_k is not None and annotation != "cluster":
            warnings.warn(
                f"Parameter kmeans_k ignored for annotation setting {annotation}."
            )

        if annotation == "population":
            self._plot_populations(pcx=pcx, pcy=pcy, fig=fig, **kwargs)
        elif annotation == "cluster":
            fig = self._plot_clusters(
                pcx=pcx, pcy=pcy, kmeans_k=kmeans_k, fig=fig, **kwargs
            )
        elif annotation is None:
            fig.add_trace(
                go.Scatter(
                    x=self.pca_data[f"PC{pcx}"],
                    y=self.pca_data[f"PC{pcy}"],
                    mode="markers",
                    marker_color=marker_color,
                    name=name,
                    **kwargs,
                )
            )
        else:
            raise ValueError(
                f"Unrecognized annotation option {annotation}. "
                f"Allowed options are None, 'population', and 'cluster'."
            )

        xtitle = f"PC {pcx + 1}"
        ytitle = f"PC {pcy + 1}"

        if show_variance_in_axes:
            xtitle += f" ({round(self.explained_variances[pcx] * 100, 1)}%)"
            ytitle += f" ({round(self.explained_variances[pcy] * 100, 1)}%)"

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle)
        fig.update_layout(template="plotly_white", height=1000, width=1000)

        if outfile is not None:
            fig.write_image(outfile)

        return fig


def from_smartpca(smartpca_evec_file: FilePath, smartpca_eval_file: FilePath) -> PCA:
    """
    Creates a PCA object from a smartPCA results file.

    Args:
        smartpca_evec_file (FilePath): FilePath to the evec results of SmartPCA run.
        smartpca_evec_file (FilePath): FilePath to the eval results of SmartPCA run.

    Returns:
        PCA: PCA object encapsulating the results of the SmartPCA run.

    """
    # First, read the eigenvectors and transform it into the pca_data pandas dataframe
    with open(smartpca_evec_file) as f:
        # first line does not contain data we are interested in
        f.readline()
        pca_data = pd.read_table(f, delimiter=" ", skipinitialspace=True, header=None)

    n_pcs = pca_data.shape[1] - 2

    cols = ["sample_id", *[f"PC{i}" for i in range(n_pcs)], "population"]
    pca_data = pca_data.rename(columns=dict(zip(pca_data.columns, cols)))
    pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

    # next, read the eigenvalues and compute the explained variances for all n_pcs principal components
    eigenvalues = open(smartpca_eval_file).readlines()
    eigenvalues = [float(ev) for ev in eigenvalues]
    explained_variances = [ev / sum(eigenvalues) for ev in eigenvalues]
    # keep only the first n_pcs explained variances
    explained_variances = explained_variances[:n_pcs]

    return PCA(pca_data=pca_data, explained_variances=explained_variances, n_pcs=n_pcs)
