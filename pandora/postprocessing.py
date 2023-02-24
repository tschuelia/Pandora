import numpy as np
import pandas as pd

from pandora.custom_types import *


def read_smartpca_eigenvec(filename: FilePath) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(filename) as f:
        # first, read the eigenvalues and compute the explained variances
        # the first line looks like this:
        #  #eigvals: 124.570 78.762    ...
        eigenvalues = f.readline().split()[1:]
        eigenvalues = [float(ev) for ev in eigenvalues]
        explained_variances = [(ev, ev / sum(eigenvalues)) for ev in eigenvalues]
        explained_variances = pd.DataFrame(explained_variances, columns=["eigenvalue", "explained_variance"])

        # next, read the PCs per sample
        df = pd.read_table(f, delimiter=" ", skipinitialspace=True, header=None)
    n_pcs = df.shape[1] - 2
    cols = ["sample_id"] + [f"PC{i}" for i in range(n_pcs)] + ["population"]
    df = df.rename(columns=dict(zip(df.columns,cols)))
    return df, explained_variances


def clean_converted_names(evecs: pd.DataFrame) -> pd.DataFrame:
    names = []
    for idx, row in evecs.iterrows():
        _, name = row.sample_id.split(":")
        names.append(name)

    evecs.sample_id = names
    return evecs


def annotate_population(empirical: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    populations = []
    for idx, bootstrap_row in bootstrap.iterrows():
        empirical_row = empirical.loc[empirical.sample_id == bootstrap_row.sample_id]
        assert empirical_row.shape[0] == 1, empirical_row.shape[0]

        populations.append(empirical_row.population.item())

    bootstrap["population"] = populations
    return bootstrap
