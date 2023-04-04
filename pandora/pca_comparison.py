from scipy.spatial import procrustes
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import euclidean_distances

from pandora.custom_types import *
from pandora.pca import PCA


def _correct_missing(pca: PCA, samples_in_both):
    pca_data = pca.pca_data
    pca_data = pca_data.loc[pca_data.sample_id.isin(samples_in_both)]

    return PCA(
        pca_data=pca_data, explained_variances=pca.explained_variances, n_pcs=pca.n_pcs
    )


class PCAComparison:
    def __init__(self, comparable: PCA, reference: PCA):
        self.comparable = comparable
        self.reference = reference

        self._clip_missing_samples_for_comparison()

    def _clip_missing_samples_for_comparison(self) -> None:
        comp_data = self.comparable.pca_data
        ref_data = self.reference.pca_data

        comp_ids = set(comp_data.sample_id)
        ref_ids = set(ref_data.sample_id)

        self.shared_samples = sorted(comp_ids.intersection(ref_ids))

        self.comparable_clipped = _correct_missing(self.comparable, self.shared_samples)
        self.reference_clipped = _correct_missing(self.reference, self.shared_samples)

        assert (
            self.comparable_clipped.pc_vectors.shape
            == self.reference_clipped.pc_vectors.shape
        )

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
        comp_data = self.comparable_clipped.pc_vectors
        ref_data = self.reference_clipped.pc_vectors

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
            n_clusters = self.reference_clipped.get_optimal_n_clusters()

        # since we are only comparing the assigned cluster labels, we don't need to transform self prior to comparing
        comp_kmeans = self.comparable_clipped.cluster(
            n_clusters=n_clusters, weighted=weighted
        )
        ref_kmeans = self.reference_clipped.cluster(
            n_clusters=n_clusters, weighted=weighted
        )

        comp_cluster_labels = comp_kmeans.predict(self.comparable_clipped.pc_vectors)
        ref_cluster_labels = ref_kmeans.predict(self.reference_clipped.pc_vectors)

        return fowlkes_mallows_score(ref_cluster_labels, comp_cluster_labels)

    def detect_rogue_samples(self, rogue_cutoff: float) -> List[str]:
        """
        Returns a list of sample IDs that are considered rogue samples when comparing self to other.
        A sample is considered rogue if the euclidean distance between its PC vectors in self and other
        is larger than the rogue_cutoff-percentile of pairwise PC vector distances
        """
        # make sure we are comparing the correct PC-vectors in the following
        assert np.all(
            self.comparable_clipped.pca_data.sample_id
            == self.reference_clipped.pca_data.sample_id
        )

        sample_distances = euclidean_distances(
            self.reference_clipped.pc_vectors, self.comparable_clipped.pc_vectors
        ).diagonal()
        sample_distances = pd.Series(sample_distances)
        rogue_threshold = sample_distances.quantile(rogue_cutoff)

        rogue_samples = [
            sample_id
            for dist, sample_id in zip(sample_distances, self.shared_samples)
            if dist > rogue_threshold
        ]

        """
        TODO: die Idee hier ist jetzt, dass man von den sample distances vielleicht das
        95% oder 99% quantile nimmt und alle samples filtert die eine distanz dadrüber haben
        dann kann man schauen wie oft welches sample bei wie vielen bootstraps als rogue identifiziert wird
        und der support wert für das sample ist dann 1 - (# rogue / # bootstraps) oder so 
        """
        return rogue_samples
