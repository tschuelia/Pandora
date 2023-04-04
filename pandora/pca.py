from __future__ import annotations  # allows type hint PCA inside PCA class

import warnings

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pandora.custom_types import *


def _get_colors(n: int) -> List[str]:
    """Returns a list of n HSV colors evenly spaced in the HSV colorspace.

    Args:
        n (int): Number of colors to return

    Returns:
        List[str]: List of n plotly HSV color strings.

    """
    return [f"hsv({v}%, 100%, 80%)" for v in np.linspace(0, 100, n, endpoint=False)]


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

    def get_optimal_n_clusters(self, min_n: int = 3, max_n: int = 50) -> int:
        """
        Determines the optimal number of clusters k for K-Means clustering.

        Args:
            min_n (int): Minimum number of clusters. Defaults to 3.
            max_n (int): Maximum number of clusters. Defaults to 50.

        Returns:
            int: the optimal number of clusters between min_n and max_n
        """
        best_k = -1
        best_score = -1

        for k in range(min_n, max_n):
            # TODO: what range is reasonable?
            kmeans = KMeans(random_state=42, n_clusters=k, n_init=10)
            kmeans.fit(self.pc_vectors)
            score = silhouette_score(self.pc_vectors, kmeans.labels_)
            best_k = k if score > best_score else best_k
            best_score = max(score, best_score)

        return best_k

    def cluster(self, n_clusters: int = None, weighted: bool = True) -> KMeans:
        """
        Fits a K-Means cluster to the pca data and returns a scikit-learn fitted KMeans object.

        Args:
            n_clusters (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.
            weighted (bool): If set, scales the PCA data of self and other according to the respective explained variances prior to clustering.

        Returns:
            KMeans: Scikit-learn KMeans object that with n_clusters that is fitted to self.pca_data.
        """
        pca_data_np = self.pc_vectors
        if n_clusters is None:
            n_clusters = self.get_optimal_n_clusters()
        if weighted:
            pca_data_np = pca_data_np * self.explained_variances
        kmeans = KMeans(random_state=42, n_clusters=n_clusters, n_init=10)
        kmeans.fit(pca_data_np)
        return kmeans

    def plot(
        self,
        pc1: int = 0,
        pc2: int = 1,
        annotation: str = None,
        n_clusters: int = None,
        fig: go.Figure = None,
        marker_color="darkseagreen",
        name: str = "",
        outfile: FilePath = None,
        redo: bool = False,
        **kwargs,
    ) -> go.Figure:
        """
        Plots the PCA data for pc1 and pc2.
        TODO: update args description

        Args:
            pc1 (int): Number of the PC to plot on the x-axis.
                The PCs are 0-indexed, so to plot the first principal component, set pc1 = 0. Defaults to 0.
            pc2 (int): Number of the PC to plot on the y-axis.
                The PCs are 0-indexed, so to plot the second principal component, set pc2 = 1. Defaults to 1.
            annotation (bool): If None, plots alls samples with the same color.
                If "population", plots each population with a different color.
                If "cluster", applies K-Means clustering and plots each cluster with a different color.
            n_clusters (int): TODO
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

        if n_clusters is not None and annotation != "cluster":
            warnings.warn(
                f"Parameter n_clusters ignored for annotation setting {annotation}."
            )

        if annotation == "population":
            if self.pca_data.population.isna().all():
                raise ValueError(
                    "Cannot plot populations: no populations associated with PCA data."
                )

            populations = self.pca_data.population.unique()
            colors = _get_colors(len(populations))

            assert len(populations) == len(colors), f"{len(populations)}, {len(colors)}"

            for i, population in enumerate(populations):
                _data = self.pca_data.loc[self.pca_data.population == population]
                fig.add_trace(
                    go.Scatter(
                        x=_data[f"PC{pc1}"],
                        y=_data[f"PC{pc2}"],
                        mode="markers",
                        marker_color=colors[i],
                        name=population,
                        **kwargs,
                    )
                )
        elif annotation == "cluster":
            n_clusters = (
                n_clusters if n_clusters is not None else self.get_optimal_n_clusters()
            )
            cluster_labels = self.cluster(n_clusters=n_clusters).labels_

            _pca_data = self.pca_data.copy()
            _pca_data["cluster"] = cluster_labels

            colors = _get_colors(n_clusters)

            for i in range(n_clusters):
                _data = _pca_data.loc[_pca_data.cluster == i]
                fig.add_trace(
                    go.Scatter(
                        x=_data[f"PC{pc1}"],
                        y=_data[f"PC{pc2}"],
                        mode="markers",
                        marker_color=colors[i],
                        name=f"Cluster {i + 1}",
                        **kwargs,
                    )
                )

        elif annotation is None:
            fig.add_trace(
                go.Scatter(
                    x=self.pca_data[f"PC{pc1}"],
                    y=self.pca_data[f"PC{pc2}"],
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

        xtitle = f"PC {pc1 + 1}"
        ytitle = f"PC {pc2 + 1}"

        if show_variance_in_axes:
            xtitle += f" ({round(self.explained_variances[pc1] * 100, 1)}%)"
            ytitle += f" ({round(self.explained_variances[pc2] * 100, 1)}%)"

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle)
        fig.update_layout(template="plotly_white", height=1000, width=1000)

        if outfile is not None:
            fig.write_image(outfile)

        return fig


def transform_pca_to_reference(pca: PCA, pca_reference: PCA) -> Tuple[PCA, PCA]:
    """
    Finds a transformation matrix that most closely matches pca to pca_reference and transforms pca.
    Both PCAs are standardized prior to transformation.

    Args:
        pca (PCA): The PCA that should be transformed
        pca_reference (PCA): The PCA that pca1 should be transformed towards

    Returns:
        Tuple[PCA, PCA]: Two new PCA objects, the first one is the transformed pca and the second one is the standardized pca_reference.
            In all downstream comparisons or pairwise plotting, use these PCA objects.
    """

    # TODO: check whether the sample IDs match -> we can only compare PCAs for the same samples
    pca_data = pca.pc_vectors
    pca_ref_data = pca_reference.pc_vectors

    if pca_data.shape[0] != pca_ref_data.shape[0]:
        # mismatch in sample_ids, impute data by adding zero-vectors
        _pca, _pca_ref = _clip_missing_samples_for_comparison(pca, pca_reference)
        pca_data = _pca.pc_vectors
        pca_ref_data = _pca_ref.pc_vectors

    if pca_data.shape != pca_ref_data.shape:
        raise ValueError(
            "Mismatch in PCA size: PCA1 and PCA2 need to have the same number of PCs."
        )

    # TODO: reorder PCs (if we find a dataset where this is needed...don't want to blindly implement something)
    standardized_reference, transformed_pca, _ = procrustes(pca_ref_data, pca_data)

    standardized_reference = PCA(
        pca_data=standardized_reference,
        explained_variances=pca_reference.explained_variances,
        n_pcs=pca_reference.n_pcs,
        sample_ids=pca_reference.pca_data.sample_id,
        populations=pca_reference.pca_data.population
    )

    transformed_pca = PCA(
        pca_data=transformed_pca,
        explained_variances=pca.explained_variances,
        n_pcs=pca.n_pcs,
        sample_ids=pca.pca_data.sample_id,
        populations=pca.pca_data.population,
    )

    return standardized_reference, transformed_pca


def from_smartpca(smartpca_evec_file: FilePath) -> PCA:
    """
    Creates a PCA object from a smartPCA results file.

    Args:
        smartpca_evec_file (FilePath): FilePath to the results of SmartPCA run.

    Returns:
        PCA: PCA object encapsulating the results of the SmartPCA run.

    """
    with open(smartpca_evec_file) as f:
        # first, read the eigenvalues and compute the explained variances
        # the first line looks like this:
        #  #eigvals: 124.570 78.762    ...
        eigenvalues = f.readline().split()[1:]
        eigenvalues = [float(ev) for ev in eigenvalues]
        explained_variances = [ev / sum(eigenvalues) for ev in eigenvalues]

        # next, read the PCs per sample
        pca_data = pd.read_table(f, delimiter=" ", skipinitialspace=True, header=None)

    n_pcs = pca_data.shape[1] - 2

    cols = ["sample_id", *[f"PC{i}" for i in range(n_pcs)], "population"]
    pca_data = pca_data.rename(columns=dict(zip(pca_data.columns, cols)))
    pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

    return PCA(pca_data=pca_data, explained_variances=explained_variances, n_pcs=n_pcs)


def from_plink(plink_evec_file: FilePath, plink_eval_file: FilePath) -> PCA:
    # read the eigenvalues
    eigenvalues = [float(ev.strip()) for ev in open(plink_eval_file)]
    explained_variances = [ev / sum(eigenvalues) for ev in eigenvalues]

    n_pcs = len(explained_variances)
    cols = ["sample_id", *[f"PC{i}" for i in range(n_pcs)]]

    with open(plink_evec_file) as f:
        # first, read the eigenvalues
        # the first line is the header, we are going to ignore this
        f.readline()

        # next, read the PCs per sample
        pca_data = pd.read_table(f, delimiter="\t", skipinitialspace=False, header=None, names=cols)

    pca_data = pca_data.rename(columns=dict(zip(pca_data.columns, cols)))
    pca_data = pca_data.sort_values(by="sample_id").reset_index(drop=True)

    return PCA(pca_data=pca_data, explained_variances=explained_variances, n_pcs=n_pcs)


def from_sklearn(
    evec_file: FilePath,
    eval_file: FilePath,
    plink_id_file: FilePath
) -> PCA:
    sample_ids = [l.strip() for l in plink_id_file.open().readlines()[1:]]
    pca_data = np.load(evec_file)
    explained_variances = np.load(eval_file)

    return PCA(
        pca_data=pca_data,
        explained_variances=explained_variances,
        n_pcs=explained_variances.shape[0],
        sample_ids=sample_ids
    )