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
### Command line interface
Not yet implemented, coming soon...

### From code
Once installed, you can use Pandora from code, e.g. to compare your own PCAs:

```python
import numpy as np
from pandora.comparison import match_pcas, compare_clustering
from pandora.postprocessing import smartpca_results_to_dataframes

# pca1 and pca2 can be either pandas dataframes or numpy arrays
pca1 = ...
pca2 = ...

# for example, you can load a smartpca result using read_smartpca_eigenvec
pca1 = smartpca_results_to_dataframes("path/to/smartpca.evec")

# find a transformation matrix that matches pca2 to pca1, and then applies this transformation to pca2
pca1, transformed_pca2, transformation = match_pcas(pca1, pca2)

# get a set of cluster metrics to compare the clustering of pca1 and pca2
scores = compare_clustering(pca1, pca2)
```

## How Pandora works:
### Bootstrapping:
Bootstrapping the original data is done on two levels independently: resampling the SNPs and resampling the individual sequences. 
Bootstrapping the SNPs is straightforward and has a clear interpretation. This is not the case for resampling the sequences.
TODO: describe how we resample the sequences and what this means (refer to the PCA Nature paper)

### PCA uncertainty
To measure the uncertainty of the dimensionality reduction, we compare the PCA of the original dataset (`original_pca`) to the PCA of each bootstrapped dataset (`bootstrapped_pca_i`) separately.

