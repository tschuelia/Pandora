# Pandora
Pandora's Box for PCA analyses. This tool is intended to quantify the uncertainty of PCA analyses. 
As the name suggests: there might be results surfacing that you don't like :-)

Pandora estimates the uncertainty of PCA dimensionality reduction using bootstrap resampling on two levels: SNP level and individual level.
Pandora then reports the uncertainty on two levels:

**PCA level:** 
On a scale of 0 – 1, how closely do the resampled PCAs match the PCA on the original data. 
0 is the worst possible score and 1 the best possible score where all bootstrapped PCAs are exactly identical to the original PCA.
The lower the score, the more careful you should be to draw conclusions based on the PCA of the data.

**Cluster level:**
Typically, a PCA analysis is followed by a clustering (usually using K-Means clustering) to determine e.g. population structures in the data.
Pandora estimates the uncertainty of the clustering using the bootstrapped PCAs and the TODO score. 
Again, the uncertainty is measured on a scale of 0 to 1, with 0 being the worst possible score and 1 the best.

## Installation
I recommend to install Pandora in a (conda) environment.

```commandline
git clone https://github.com/tschuelia/Pandora.git
cd Pandora
pip install -e .
```

The `-e` flag is a development flag I would recommend to set. Changes in the code will be reflected in the respective package and a reinstallation is not necessary.
On some machines this option does not work. If you encounter any errors, try again using `pip install .`

## Usage
### Comparing two PCAs
E.g. comparing two different smartPCA results.
The number of PCs in both PCs must be identical. The number of samples can vary, Pandora will take care of that.
```python
from pandora.pca import from_smartpca
from pandora.pca_comparison import PCAComparison

pca1 = from_smartpca("path/to/smart.pca1.evec")  # this is a PCA object, see pandora.pca::PCA for more details
pca2 = from_smartpca("path/to/smart.pca2.evec")  # this is a PCA object, see pandora.pca::PCA for more details

comparison = PCAComparison(pca1, pca2)  # this is a PCAComparison object, see pandora.pca_comparison::PCA for more details
similarity = comparison.compare()
print("PCA1 and PCA2 similarity: ", similarity)

kmeans_cluster_similarity = comparison.compare_clustering()
print("PCA1 and PCA2 similarity of K-Means clustering: ", kmeans_cluster_similarity)

# plot each PCA individually using pca.plot:
# outfile is a path where to write the figure to
pca1.plot(outfile="path/to/pca1_fig.pdf")
pca2.plot(outfile="path/to/pca2_fig.pdf")

# plot pca1 and pca2 jointly
comparison.plot(outfile="path/to/both.pdf")

# Experimental Feature: plot rogue samples = samples with high distances when comparing the PC-vectors
comparison.plot(show_rogue=True, outfile="path/to/both_with_rogue.pdf")
```
### Input
The following input types are currently supported:
- EIGEN format

Pandora applies no preprocessing so make sure to provide a preprocessed dataset. More specifically, apply LD pruning *before* running Pandora.

### Command line interface
Not yet implemented, coming soon...

## How Pandora works:
### Bootstrapping:
We bootstrap the original data by resampling the SNPs. TODO: describe the bootstrapping procedure.

### PCA uncertainty
To measure the uncertainty of the dimensionality reduction, we compare the PCA of the original dataset (`original_pca`) to the PCA of each bootstrapped dataset (`bootstrapped_pca_i`) separately.

