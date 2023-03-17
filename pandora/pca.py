from __future__ import annotations  # allows type hint PCA inside PCA class

import math
import tempfile
import subprocess
import textwrap
import warnings

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.colors import color_parser, label_rgb, unlabel_rgb

from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    fowlkes_mallows_score,
)
from sklearn.decomposition import PCA as sklearnPCA

from pandora.custom_types import *
from pandora.logger import logger, fmt_message


def _color_space(lowcolor: str, highcolor: str, n_colors: int, endpoint: bool = True):
    lowcolor = unlabel_rgb(lowcolor)
    highcolor = unlabel_rgb(highcolor)

    lowred, lowblue, lowgreen = lowcolor
    highred, highblue, highgreen = highcolor

    r_values = np.linspace(lowred, highred, n_colors, endpoint=endpoint).clip(0, 255)
    b_values = np.linspace(lowblue, highblue, n_colors, endpoint=endpoint).clip(0, 255)
    g_values = np.linspace(lowgreen, highgreen, n_colors, endpoint=endpoint).clip(0, 255)

    colors = zip(r_values, b_values, g_values)

    return color_parser(colors, label_rgb)


def _get_colors(n: int) -> List[str]:
    """Returns a list of n RGB colors evenly spaced in the red – green – blue space.

    Args:
        n (int): Number of colors to return

    Returns:
        List[str]: List of n plotly RGB color strings.

    """
    if n <= 3:
        return ["rgb(255, 0, 0)", "rgb(0,255,0)", "rgb(0,0,255)"][:n]

    red_green = _color_space(
        lowcolor="rgb(255,0,0)",
        highcolor="rgb(0,255,0)",
        n_colors=n // 2,
        endpoint=False
    )
    green_blue = _color_space(
        lowcolor="rgb(0,255,0)",
        highcolor="rgb(0,0,255)",
        n_colors=math.ceil(n / 2),
        endpoint=True
    )
    return red_green + green_blue


def _correct_for_missing_samples(pca: PCA, samples_to_add: set) -> PCA:
    if len(samples_to_add) == 0:
        return pca
    # TODO: very ugly code, refactor!
    columns = [c for c in pca.pca_data.columns if "PC" in c]
    imputation_data = list(zip(columns, [0.0] * pca.n_pcs))

    new_data = []
    for s in samples_to_add:
        data = [("sample_id", s)] + imputation_data
        new_data.append(pd.DataFrame(dict(data), index=[0]))

    new_data = pd.concat(new_data)
    pca_data = pd.concat([pca.pca_data, new_data], ignore_index=True).reset_index(drop=True)
    return PCA(
        pca_data=pca_data,
        explained_variances=pca.explained_variances,
        n_pcs=pca.n_pcs
    )


def _impute_missing_samples_for_comparison(pca1: PCA, pca2: PCA) -> Tuple[PCA, PCA]:
    pca1_data = pca1.pca_data
    pca2_data = pca2.pca_data

    pca1_ids = set(pca1_data.sample_id)
    pca2_ids = set(pca2_data.sample_id)

    pca1 = _correct_for_missing_samples(pca1, pca2_ids - pca1_ids)
    pca2 = _correct_for_missing_samples(pca2, pca1_ids - pca2_ids)

    assert pca1.pc_vectors.shape == pca2.pc_vectors.shape

    return pca1, pca2


def _correct_missing(pca: PCA, samples_in_both):
    pca_data = pca.pca_data
    pca_data = pca_data.loc[pca_data.sample_id.isin(samples_in_both)]

    return PCA(
        pca_data=pca_data,
        explained_variances=pca.explained_variances,
        n_pcs=pca.n_pcs
    )


def _clip_missing_samples_for_comparison(pca1: PCA, pca2: PCA) -> Tuple[PCA, PCA]:
    pca1_data = pca1.pca_data
    pca2_data = pca2.pca_data

    pca1_ids = set(pca1_data.sample_id)
    pca2_ids = set(pca2_data.sample_id)

    in_both = pca1_ids.intersection(pca2_ids)

    pca1 = _correct_missing(pca1, in_both)
    pca2 = _correct_missing(pca2, in_both)

    assert pca1.pc_vectors.shape == pca2.pc_vectors.shape

    return pca1, pca2



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

    def compare(self, other: PCA) -> float:
        """
        Compare self to other by transforming self towards other and then computing the samplewise cosine similarity.
        Returns the average and standard deviation. The resulting similarity is on a scale of 0 to 1, with 1 meaning
        self and other are identical.
        TODO: nope wir nehmen jetzt doch procrustes und die similarity von procrustes direkt

        Args:
            other (PCA): PCA object to compare self to.

        Returns:
            float: Similarity as average cosine similarity per sample PC-vector in self and other.
        """
        # TODO: check whether the sample IDs match -> we can only compare PCAs for the same samples

        # check if the number of samples match for now
        self_data = self.pc_vectors
        other_data = other.pc_vectors

        if self_data.shape[0] != other_data.shape[0]:
            # mismatch in sample_ids, impute data by adding zero-vectors
            # _self, _other = _impute_missing_samples_for_comparison(self, other)
            _self, _other = _clip_missing_samples_for_comparison(self, other)
            self_data = _self.pc_vectors
            other_data = _other.pc_vectors

        _, _, disparity = procrustes(self_data, other_data)
        similarity = np.sqrt(1 - disparity)

        return similarity

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

    def compare_clustering(
        self, other: PCA, n_clusters: int = None, weighted: bool = True
    ) -> float:
        """
        Compare self clustering to other clustering using other as ground truth.

        Args:
            other (PCA): PCA object to compare self to.
            n_clusters (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.
            weighted (bool): If set, scales the PCA data of self and other according to the respective explained variances prior to clustering.

        Returns:
            float: The Fowlkes-Mallow score of Cluster similarity between the clusters of self and other
        """
        if n_clusters is None:
            # we are comparing self to other -> use other as ground truth
            # thus, we determine the number of clusters using other
            n_clusters = other.get_optimal_n_clusters()

        if self.pc_vectors.shape[0] != other.pc_vectors.shape[0]:
            # mismatch in sample_ids, impute data by adding zero-vectors
            # _self, _other = _impute_missing_samples_for_comparison(self, other)
            _self, _other = _clip_missing_samples_for_comparison(self, other)
        else:
            _self = self
            _other = other

        # since we are only comparing the assigned cluster labels, we don't need to transform self prior to comparing
        self_kmeans = _self.cluster(n_clusters=n_clusters, weighted=weighted)
        other_kmeans = _other.cluster(n_clusters=n_clusters, weighted=weighted)

        self_cluster_labels = self_kmeans.predict(_self.pc_vectors)
        other_cluster_labels = other_kmeans.predict(_other.pc_vectors)

        return fowlkes_mallows_score(other_cluster_labels, self_cluster_labels)

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
            fig (go.Figure): If set, appends the PCA data to this fig. Default is to plot on a new, empty figure.
            name (str): Name of the trace in the resulting plot. Setting a name will only have an effect if
                plot_populations = False and fig is not None.

        Returns:
            go.Figure: Plotly figure containing a scatter plot of the PCA data.
        """
        if not fig:
            fig = go.Figure()

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

        xtitle = f"PC {pc1 + 1} ({round(self.explained_variances[pc1] * 100, 1)}%)"
        ytitle = f"PC {pc2 + 1} ({round(self.explained_variances[pc2] * 100, 1)}%)"

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
        # _pca, _pca_ref = _impute_missing_samples_for_comparison(pca, pca_reference)
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


def run_smartpca(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    n_pcs: int = 20,
    redo: bool = False,
) -> PCA:
    geno = pathlib.Path(f"{infile_prefix}.geno")
    snp = pathlib.Path(f"{infile_prefix}.snp")
    ind = pathlib.Path(f"{infile_prefix}.ind")

    files_exist = all([geno.exists(), snp.exists(), ind.exists()])
    if not files_exist:
        raise ValueError(
            f"Not all input files for file prefix {infile_prefix} present. "
            f"Looking for files in EIGEN format with endings .geno, .snp, and .ind"
        )

    evec_out = pathlib.Path(f"{outfile_prefix}.evec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eval")
    smartpca_log = pathlib.Path(f"{outfile_prefix}.smartpca.log")

    files_exist = all([evec_out.exists(), eval_out.exists(), smartpca_log.exists()])

    if files_exist and not redo:
        # TODO: das reicht nicht als check, bei unfertigen runs sind die files einfach nicht vollständig aber
        #  leider noch vorhanden
        logger.info(
            fmt_message(f"Skipping smartpca. Files {outfile_prefix}.* already exist.")
        )
        return from_smartpca(evec_out)

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        _df = pd.read_table(
            f"{infile_prefix}.ind", delimiter=" ", skipinitialspace=True, header=None
        )
        num_populations = _df[2].unique().shape[0]

        conversion_content = f"""
            genotypename: {infile_prefix}.geno
            snpname: {infile_prefix}.snp
            indivname: {infile_prefix}.ind
            evecoutname: {evec_out}
            evaloutname: {eval_out}
            numoutevec: {n_pcs}
            maxpops: {num_populations}
            """
        #numoutlieriter: 0
        projection_file = pathlib.Path(f"{infile_prefix}.population")
        if projection_file.exists():
            conversion_content += f"\npoplistname: {projection_file}"

        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            smartpca,
            "-p",
            tmpfile.name,
        ]
        with smartpca_log.open("w") as logfile:
            subprocess.run(cmd, stdout=logfile, stderr=logfile)

    return from_smartpca(evec_out)


def check_pcs_sufficient(explained_variances: List, cutoff: float) -> Union[int, None]:
    # if all PCs explain more than 1 - <cutoff>% variance we consider the number of PCs to be insufficient
    if all([e > (1 - cutoff) for e in explained_variances]):
        return

    # otherwise, find the index of the last PC explaining more than <cutoff>%
    sum_variances = 0
    for i, var in enumerate(explained_variances, start=1):
        sum_variances += var
        if sum_variances >= cutoff:
            logger.info(
                fmt_message(
                    f"Optimal number of PCs for explained variance cutoff {cutoff}: {i}"
                )
            )
            return i


def determine_number_of_pcs(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    explained_variance_cutoff: float = 0.95,
    redo: bool = False,
) -> int:
    n_pcs = 20
    pca_checkpoint = pathlib.Path(f"{outfile_prefix}.ckp")

    if pca_checkpoint.exists() and not redo:
        # checkpointing file contains three values: an int, a bool, and a float
        # the int is the number of PCs that was last tested
        # the bool says whether the analysis finished properly or not
        # the float (ignored here) is the amount of variance explained by the current number of PCs (used for debugging)
        n_pcs, finished = pca_checkpoint.open().readline().strip().split()
        n_pcs = int(n_pcs)
        finished = bool(int(finished))

        if finished:
            logger.info(
                fmt_message(
                    f"Resuming from checkpoint: determining number of PCs already finished."
                )
            )

            # check if the last smartpca run already had the optimal number of PCs present
            # if yes, create a new PCA object and truncate the data to the optimal number of PCs
            return n_pcs

        # otherwise, running smartPCA was aborted and did not finnish properly, resume from last tested n_pcs
        logger.info(
            fmt_message(
                f"Resuming from checkpoint: Previously tested setting {n_pcs} not sufficient. "
                f"Repeating with n_pcs = {int(1.5 * n_pcs)}"
            )
        )
        n_pcs = int(1.5 * n_pcs)
    else:
        logger.info(
            fmt_message(
                f"Determining number of PCs. Now running PCA analysis with {n_pcs} PCs."
            )
        )

    while True:
        with tempfile.TemporaryDirectory() as tmp_outdir:
            tmp_outfile_prefix = pathlib.Path(tmp_outdir) / "determine_npcs"
            pca = run_smartpca(
                infile_prefix=infile_prefix,
                outfile_prefix=tmp_outfile_prefix,
                smartpca=smartpca,
                n_pcs=n_pcs,
                redo=True,
            )
            best_pcs = check_pcs_sufficient(
                pca.explained_variances, explained_variance_cutoff
            )
            if best_pcs:
                pca_checkpoint.write_text(f"{best_pcs} 1")
                return best_pcs

        # if all PCs explain >= <cutoff>% variance, rerun the PCA with an increased number of PCs
        # we increase the number by a factor of 1.5
        logger.info(
            fmt_message(
                f"{n_pcs} PCs not sufficient. Repeating analysis with {int(1.5 * n_pcs)} PCs."
            )
        )
        # number of PCs not sufficient, write checkpoint and increase n_pcs
        pca_checkpoint.write_text(f"{n_pcs} 0 {sum(pca.explained_variances)}")
        n_pcs = int(1.5 * n_pcs)


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


def run_plink(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    plink: Executable,
    n_pcs: int = 20,
    redo: bool = False,
) -> PCA:
    evec_out = pathlib.Path(f"{outfile_prefix}.eigenvec")
    eval_out = pathlib.Path(f"{outfile_prefix}.eigenval")
    plink_log = pathlib.Path(f"{outfile_prefix}.plinkpca.log")

    files_exist = all([evec_out.exists(), eval_out.exists(), plink_log.exists()])

    if files_exist and not redo:
        # TODO: das reicht nicht als check, bei unfertigen runs sind die files einfach nicht vollständig aber leider noch vorhanden
        logger.info(
            fmt_message(f"Skipping plink PCA. Files {outfile_prefix}.* already exist.")
        )
        return from_plink(evec_out, eval_out)

    pca_cmd = [
        plink,
        "--pca",
        str(n_pcs),
        "--bfile",
        infile_prefix,
        "--out",
        outfile_prefix,
        "--no-fid"
    ]

    with plink_log.open("w") as logfile:
        subprocess.run(pca_cmd, stdout=logfile, stderr=logfile)

    return from_plink(evec_out, eval_out)


def run_sklearn(
    outfile_prefix: FilePath,
    n_pcs: int = 20,
    redo: bool = False,
) -> PCA:
    plink_snp_data = pathlib.Path(f"{outfile_prefix}.rel")
    plink_sample_data = pathlib.Path(f"{outfile_prefix}.rel.id")

    pc_vectors_file = pathlib.Path(f"{outfile_prefix}.sklearn.evec.npy")
    variances_file = pathlib.Path(f"{outfile_prefix}.sklearn.eval.npy")

    if redo or (not pc_vectors_file.exists() and not variances_file.exists()):
        snp_data = []
        for line in plink_snp_data.open():
            values = line.split()
            values = [float(v) for v in values]
            snp_data.append(values)

        snp_data = np.asarray(snp_data)

        pca = sklearnPCA(n_components=n_pcs)
        pca_data = pca.fit_transform(snp_data)

        np.save(pc_vectors_file, pca_data)
        np.save(variances_file, pca.explained_variance_ratio_)

    return from_sklearn(
        evec_file=pc_vectors_file,
        eval_file=variances_file,
        plink_id_file=plink_sample_data
    )


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