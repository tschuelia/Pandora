from scipy.spatial import procrustes
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import euclidean_distances

from pandora.custom_types import *
from pandora.pca import PCA
from pandora.utils import get_colors


def _correct_missing(pca: PCA, samples_in_both):
    pca_data = pca.pca_data
    pca_data = pca_data.loc[pca_data.sample_id.isin(samples_in_both)]

    return PCA(
        pca_data=pca_data, explained_variances=pca.explained_variances, n_pcs=pca.n_pcs
    )


def _clip_missing_samples_for_comparison(comparable: PCA, reference: PCA) -> Tuple[PCA, PCA]:
    comp_data = comparable.pca_data
    ref_data = reference.pca_data

    comp_ids = set(comp_data.sample_id)
    ref_ids = set(ref_data.sample_id)

    shared_samples = sorted(comp_ids.intersection(ref_ids))

    comparable_clipped = _correct_missing(comparable, shared_samples)
    reference_clipped = _correct_missing(reference, shared_samples)

    assert (
        comparable_clipped.pc_vectors.shape
        == reference_clipped.pc_vectors.shape
    )
    # TODO: check whether there are samples left after clipping to make sure there is something to compare...

    return comparable_clipped, reference_clipped


class PCAComparison:
    def __init__(self, comparable: PCA, reference: PCA):
        # first we transform comparable towards reference such that we can compare the PCAs
        self.comparable, self.reference = match_and_transform(comparable=comparable, reference=reference)

    def compare(self) -> float:
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
        comp_data = self.comparable.pc_vectors
        ref_data = self.reference.pc_vectors

        assert comp_data.shape == ref_data.shape

        _, _, disparity = procrustes(comp_data, ref_data)
        similarity = np.sqrt(1 - disparity)

        return similarity

    def compare_clustering(
        self, n_clusters: int = None, weighted: bool = True
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
            n_clusters = self.reference.get_optimal_n_clusters()

        # since we are only comparing the assigned cluster labels, we don't need to transform self prior to comparing
        comp_kmeans = self.comparable.cluster(
            n_clusters=n_clusters, weighted=weighted
        )
        ref_kmeans = self.reference.cluster(
            n_clusters=n_clusters, weighted=weighted
        )

        comp_cluster_labels = comp_kmeans.predict(self.comparable.pc_vectors)
        ref_cluster_labels = ref_kmeans.predict(self.reference.pc_vectors)

        return fowlkes_mallows_score(ref_cluster_labels, comp_cluster_labels)

    def detect_rogue_samples(self, rogue_cutoff: float) -> List[str]:
        """
        Returns a list of sample IDs that are considered rogue samples when comparing self to other.
        A sample is considered rogue if the euclidean distance between its PC vectors in self and other
        is larger than the rogue_cutoff-percentile of pairwise PC vector distances
        Important: make sure that the two PCAs you compare are the transformed ones, otherwise the vectors are hardly comparable
        """
        # make sure we are comparing the correct PC-vectors in the following
        assert np.all(
            self.comparable.pca_data.sample_id
            == self.reference.pca_data.sample_id
        )

        sample_distances = euclidean_distances(
            self.reference.pc_vectors, self.comparable.pc_vectors
        ).diagonal()
        sample_distances = pd.Series(sample_distances)
        rogue_threshold = sample_distances.quantile(rogue_cutoff)

        rogue_samples = [
            sample_id
            for dist, sample_id in zip(sample_distances, self.comparable.pca_data.sample_id)
            if dist > rogue_threshold
        ]

        """
        TODO: die Idee hier ist jetzt, dass man von den sample distances vielleicht das
        95% oder 99% quantile nimmt und alle samples filtert die eine distanz dadrüber haben
        dann kann man schauen wie oft welches sample bei wie vielen bootstraps als rogue identifiziert wird
        und der support wert für das sample ist dann 1 - (# rogue / # bootstraps) oder so 
        """
        return rogue_samples

    def plot(
            self,
            pc1: int = 0,
            pc2: int = 1,
            outfile: FilePath = None,
            show_rogue: bool = False,
            rogue_cutoff: float = 0.95,
            **kwargs,
             ):
        if show_rogue:
            rogue_samples = self.detect_rogue_samples(rogue_cutoff=rogue_cutoff)
            rogue_colors = dict(zip(rogue_samples, get_colors(len(rogue_samples))))

            color_reference = [rogue_colors.get(sample, "lightgrey") for sample in self.comparable.pca_data.sample_id]

            # since the sample IDs are identical for both PCAs, we can share the same list of marker colors
            color_comparable = color_reference
        else:
            color_reference = "darkblue"
            color_comparable = "orange"

        fig = self.reference.plot(
            pc1=pc1,
            pc2=pc2,
            name="Standardized reference PCA",
            marker_color=color_reference,
            **kwargs
        )

        fig = self.comparable.plot(
            fig=fig,
            pc1=pc1,
            pc2=pc2,
            name="Transformed comparable PCA",
            marker_color=color_comparable,
            marker_symbol="star",
            **kwargs
        )

        if outfile:
            fig.write_image(outfile)

        return fig


def match_and_transform(comparable: PCA, reference: PCA) -> Tuple[PCA, PCA]:
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
    comparable, reference = _clip_missing_samples_for_comparison(comparable, reference)
    comp_data = comparable.pc_vectors
    ref_data = reference.pc_vectors

    # TODO: reorder PCs (if we find a dataset where this is needed...don't want to blindly implement something)
    standardized_reference, transformed_comparable, _ = procrustes(ref_data, comp_data)

    standardized_reference = PCA(
        pca_data=standardized_reference,
        explained_variances=reference.explained_variances,
        n_pcs=reference.n_pcs,
        sample_ids=reference.pca_data.sample_id,
        populations=reference.pca_data.population
    )

    transformed_comparable = PCA(
        pca_data=transformed_comparable,
        explained_variances=comparable.explained_variances,
        n_pcs=comparable.n_pcs,
        sample_ids=comparable.pca_data.sample_id,
        populations=comparable.pca_data.population,
    )

    return standardized_reference, transformed_comparable
