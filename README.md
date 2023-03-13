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
### Input
The following input types are currently supported:
- EIGEN format

Pandora applies no preprocessing so make sure to provide a preprocessed dataset. More specifically, apply LD pruning *before* running Pandora.

### Command line interface
Not yet implemented, coming soon...

### From code
Once installed, you can use Pandora from code, e.g. to compare your own PCAs:

```python
from pandora.pca import PCA, from_smartpca
from sklearn.decomposition import PCA as sklearnPCA


# create 2 PCA objects
# you can either create one using a numpy array / pandas dataframe manually
# e.g. based on scikit-learns pca
data = ...
n_pcs = 10
sklearn_pca = sklearnPCA(n_components=n_pcs)
sklearn_pca.fit(data)
pca1 = PCA(pca_data=sklearn_pca.transform(data), explained_variances=sklearn_pca.explained_variance_ratio_, n_pcs=n_pcs)

# or, you can load a smartpca result using from_smartpca
pca2 = from_smartpca("cool_analysis.evec")

# setting pca1 as "ground truth", we match pca2 to pca1 and compute a comparison score based on the cosine similarity
score = pca2.compare(pca1)

# also, get a set of cluster metrics to compare the clustering of pca1 and pca2
cluster_scores = pca2.compare_clustering(pca1)
```

## How Pandora works:
### Bootstrapping:
We bootstrap the original data by resampling the SNPs. TODO: describe the bootstrapping procedure.

### PCA uncertainty
To measure the uncertainty of the dimensionality reduction, we compare the PCA of the original dataset (`original_pca`) to the PCA of each bootstrapped dataset (`bootstrapped_pca_i`) separately.

