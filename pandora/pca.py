import tempfile
import subprocess
import textwrap

import numpy as np
from plotly import graph_objects as go
from plotly.colors import n_colors

from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    rand_score,
    adjusted_rand_score,
    v_measure_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
)

from pandora.custom_types import *
from pandora.logger import logger, fmt_message


def _get_colors(n: int) -> List[str]:
    """Returns a list of n RGB colors evenly spaced in the red – green – blue space.

    Args:
        n (int): Number of colors to return

    Returns:
        List[str]: List of n plotly RGB color strings.

    """
    red_green = n_colors(
        lowcolor="rgb(255,0,0)",
        highcolor="rgb(0,255,0)",
        n_colors=n // 2,
        colortype="rgb",
    )
    green_blue = n_colors(
        lowcolor="rgb(0,255,0)",
        highcolor="rgb(0,0,255)",
        n_colors=round(n / 2),
        colortype="rgb",
    )
    return red_green + green_blue


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
    """
    def __init__(
        self,
        pca_data: Union[pd.DataFrame, np.ndarray],
        explained_variances: List[float],
        n_pcs: int,
    ):
        self.n_pcs = n_pcs
        self.explained_variances = explained_variances

        if isinstance(pca_data, np.ndarray):
            pca_data = pd.DataFrame(
                pca_data, columns=[f"PC{i}" for i in range(self.n_pcs)]
            )
            pca_data["sample_id"] = None
            pca_data["population"] = None

        self.pca_data = pca_data

    def cutoff_pcs(self, new_n_pcs: int):
        """
        Truncates the PCA data and explained variances to a smaller number of PCs.

        Args:
            new_n_pcs (int): new number of PCs

        Raises:
            ValueError: if new_n_pcs is greater or equal than the current n_pcs
        """
        if new_n_pcs >= self.n_pcs:
            raise ValueError(
                "New number of PCs has to be smaller than the current number."
            )
        self.n_pcs = new_n_pcs
        self.pca_data = self.pca_data[
            ["sample_id", "population", *[f"PC{i}" for i in range(new_n_pcs)]]
        ]
        self.explained_variances = self.explained_variances[:new_n_pcs]

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

    def get_pca_data_numpy(self) -> np.ndarray:
        """
        Converts the PCA data to a numpy array.

        Returns:
             np.ndarray: Array of shape (x, y) with x = number of samples and y = self.n_pcs.
                 Does not contain the sample IDs or populations.
        """
        return self.pca_data[[f"PC{i}" for i in range(self.n_pcs)]].to_numpy()

    def _transform_self_to_other(
        self, other: Self, normalize: bool = False
    ) -> np.ndarray:
        """
        Finds a transformation matrix to match self to other as close as possible (self is transformed).

        Args:
            other (PCA): PCA object to compare self to.
            normalize (bool): Whether to normalize the PCA data of both PCAs prior to computing the transformation matrix.

        Returns:
            np.ndarray: Return the transformation matrix that most closely matches self to other.
                The resulting transformation matrix has shape (n_pcs, n_pcs).

        Raises:
            ValueError: If the number of samples or the number of PCs in self and other are not identical.
        """
        if self.pca_data.shape != other.pca_data.shape:
            raise ValueError(
                "Mismatch in PCA size: Self and other need to have the same number of samples and the same number of PCs."
            )

        pca_data_self = self.get_pca_data_numpy()
        pca_data_other = other.get_pca_data_numpy()

        # TODO: reorder PCs (if we find a dataset where this is needed...don't want to blindly implement something)
        if normalize:
            pca_data_self = pca_data_self / np.linalg.norm(pca_data_self)
            pca_data_other = pca_data_other / np.linalg.norm(pca_data_other)

        transformation, _ = orthogonal_procrustes(pca_data_other, pca_data_self)
        transformed_self = pca_data_self @ transformation

        return transformed_self

    def compare(self, other: Self, normalize: bool = False) -> float:
        """
        Compare self to other by transforming self towards other and then measuring the distance.

        Args:
            other (PCA): PCA object to compare self to.
            normalize (bool): Whether to normalize the PCA data of both PCAs prior to computing the transformation matrix.

        Returns:
            float: distance between self and other
        """
        transformed_self = self._transform_self_to_other(other, normalize)

        # TODO: somehow normalize this difference to [0, 1] to reflect the degree of similarity
        other_data = other.get_pca_data_numpy()
        if normalize:
            other_data = other_data / np.linalg.norm(other_data)

        difference = np.linalg.norm(transformed_self - other_data)
        return difference

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

        pca_data_np = self.get_pca_data_numpy()

        for k in range(min_n, max_n):
            # TODO: what range is reasonable?
            kmeans = KMeans(random_state=42, n_clusters=k, n_init=10)
            kmeans.fit(pca_data_np)
            score = silhouette_score(pca_data_np, kmeans.labels_)
            best_k = k if score > best_score else best_k
            best_score = max(score, best_score)

        return best_k

    def cluster(self, n_clusters: int = None) -> KMeans:
        """
        Fits a K-Means cluster to the pca data and returns a scikit-learn fitted KMeans object.

        Args:
            n_clusters (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.

        Returns:
            KMeans: Scikit-learn KMeans object that with n_clusters that is fitted to self.pca_data.
        """
        pca_data_np = self.get_pca_data_numpy()
        if n_clusters is None:
            n_clusters = self.get_optimal_n_clusters()
        kmeans = KMeans(random_state=42, n_clusters=n_clusters, n_init=10)
        kmeans.fit(pca_data_np)
        return kmeans

    def compare_clustering(self, other: Self, n_clusters: int = None) -> Dict[str, float]:
        """
        Compare self clustering to other clustering using other as ground truth.

        Args:
            other (PCA): PCA object to compare self to.
            n_clusters (int): Number of clusters. If not set, the optimal number of clusters is determined automatically.

        Returns:
            Dict[str, float]: A dictionary containing various cluster comparison metrics.
        """
        if n_clusters is None:
            # we are comparing self to other -> use other as ground truth
            # thus, we determine the number of clusters using other
            n_clusters = other.get_optimal_n_clusters()

        # since we are only comparing the assigned cluster labels, we don't need to transform self prior to comparing

        self_kmeans = self.cluster(n_clusters=n_clusters)
        other_kmeans = other.cluster(n_clusters=n_clusters)

        self_cluster_labels = self_kmeans.predict(self.get_pca_data_numpy())
        other_cluster_labels = other_kmeans.predict(other.get_pca_data_numpy())

        scores = {
            "Random Score": rand_score(other_cluster_labels, self_cluster_labels),
            "Adjusted random score": adjusted_rand_score(
                other_cluster_labels, self_cluster_labels
            ),
            "V-Measure score": v_measure_score(
                other_cluster_labels, self_cluster_labels
            ),
            "Adjusted Mutual info score": adjusted_mutual_info_score(
                other_cluster_labels, self_cluster_labels
            ),
            "Fowlkes-Mallows-Score": fowlkes_mallows_score(
                other_cluster_labels, self_cluster_labels
            ),
        }

        return scores

    def plot(
        self,
        pc1: int = 0,
        pc2: int = 1,
        plot_populations: bool = False,
        fig: go.Figure = None,
        name: str = "",
    ) -> go.Figure:
        """
        Plots the PCA data for pc1 and pc2.

        Args:
            pc1 (int): Number of the PC to plot on the x-axis.
                The PCs are 0-indexed, so to plot the first principal component, set pc1 = 0. Defaults to 0.
            pc2 (int): Number of the PC to plot on the y-axis.
                The PCs are 0-indexed, so to plot the second principal component, set pc2 = 1. Defaults to 1.
            plot_populations (bool): If true, each population is attributed a distinct color in the resulting plot.
            fig (go.Figure): If set, appends the PCA data to this fig. Default is to plot on a new, empty figure.
            name (str): Name of the trace in the resulting plot. Setting a name will only have an effect if
                plot_populations = False and fig is not None.

        Returns:
            go.Figure: Plotly figure containing a scatter plot of the PCA data.
        """
        if not fig:
            fig = go.Figure()

        if plot_populations:
            if self.pca_data.population.isna().all():
                raise ValueError("Cannot plot populations: no populations associated with PCA data.")

            populations = self.pca_data.population.unique()
            colors = _get_colors(len(populations))

            for i, population in enumerate(populations):
                _data = self.pca_data.loc[self.pca_data.population == population]
                fig.add_trace(
                    go.Scatter(
                        x=_data[f"PC{pc1}"],
                        y=_data[f"PC{pc2}"],
                        mode="markers",
                        marker_color=colors[i],
                        name=population
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=self.pca_data[f"PC{pc1}"],
                    y=self.pca_data[f"PC{pc2}"],
                    mode="markers",
                    marker_color="darkseagreen",
                    name=name
                )
            )

        xtitle = f"PC {pc1 + 1} ({round(self.explained_variances[pc1], 1)}%)"
        ytitle = f"PC {pc2 + 1} ({round(self.explained_variances[pc2], 1)}%)"

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle)
        fig.update_layout(template="plotly_white", height=1000, width=1000)
        return fig


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

    cols = ["sample_id"] + [f"PC{i}" for i in range(n_pcs)] + ["population"]
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

    if eval_out.exists() and eval_out.exists() and not redo:
        logger.info(
            fmt_message(f"Skipping smartpca. Files {outfile_prefix}.* already exist.")
        )
        return from_smartpca(evec_out)

    with tempfile.NamedTemporaryFile(mode="w") as tmpfile:
        _df = pd.read_table(f"{infile_prefix}.ind", delimiter=" ", skipinitialspace=True, header=None)
        num_populations = _df[2].unique().shape[0]

        conversion_content = f"""
            genotypename: {infile_prefix}.geno
            snpname: {infile_prefix}.snp
            indivname: {infile_prefix}.ind
            evecoutname: {evec_out}
            evaloutname: {eval_out}
            numoutevec: {n_pcs}
            numoutlieriter: 0
            maxpops: {num_populations}
            """

        tmpfile.write(textwrap.dedent(conversion_content))
        tmpfile.flush()

        cmd = [
            smartpca,
            "-p",
            tmpfile.name,
        ]
        subprocess.check_output(cmd)

    return from_smartpca(evec_out)


def check_pcs_sufficient(explained_variances: List, cutoff: float) -> Union[int, None]:
    # at least one PC explains less than <cutoff>% variance
    # -> find the index of the last PC explaining more than <cutoff>%
    n_pcs = sum([1 for var in explained_variances if var >= cutoff])
    logger.info(
        fmt_message(
            f"Optimal number of PCs for explained variance cutoff {cutoff}: {n_pcs}"
        )
    )
    if n_pcs != len(explained_variances):
        return n_pcs


def determine_number_of_pcs(
    infile_prefix: FilePath,
    outfile_prefix: FilePath,
    smartpca: Executable,
    explained_variance_cutoff: float = 0.01,
    redo: bool = False,
):
    n_pcs = 20
    evec_file = pathlib.Path(f"{outfile_prefix}.evec")
    eval_file = pathlib.Path(f"{outfile_prefix}.eval")

    if evec_file.exists() and eval_file.exists() and not redo:
        # in case of a restarted analysis, this ensures that we correctly update the n_pcs variable below
        pca = from_smartpca(evec_file)

        logger.info(
            fmt_message(
                f"Resuming from checkpoint: "
                f"Reading data from existing PCA outfiles {outfile_prefix}.*. "
                f"Delete files or set the redo flag in case you want to rerun the PCA."
            )
        )

        # check if the last smartpca run already had the optimal number of PCs present
        # if yes, create a new PCA object and truncate the data to the optimal number of PCs
        best_pcs = check_pcs_sufficient(
            pca.explained_variances, explained_variance_cutoff
        )

        if best_pcs:
            pca.cutoff_pcs(best_pcs)
            return pca

        # otherwise, resume the search for the optimal number of PCs from the last number of PCs
        n_pcs = pca.n_pcs
        logger.info(
            fmt_message(
                f"Resuming the search for the optimal number of PCS. "
                f"Previously tested setting: {n_pcs}, new setting: {int(n_pcs * 1.5)} "
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
        pca = run_smartpca(
            infile_prefix=infile_prefix,
            outfile_prefix=outfile_prefix,
            smartpca=smartpca,
            n_pcs=n_pcs,
            redo=True,
        )
        best_pcs = check_pcs_sufficient(
            pca.explained_variances, explained_variance_cutoff
        )
        if best_pcs:
            pca.cutoff_pcs(best_pcs)
            return pca

        # if all PCs explain >= <cutoff>% variance, rerun the PCA with an increased number of PCs
        # we increase the number by a factor of 1.5
        logger.info(
            fmt_message(
                f"{n_pcs} PCs not sufficient. Repeating analysis with {int(1.5 * n_pcs)} PCs."
            )
        )
        n_pcs = int(1.5 * n_pcs)