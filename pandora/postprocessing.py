import numpy as np
import pandas as pd

from .custom_types import *


def read_smartpca_eigenvec(filename: FilePath):
    f = open(filename)

    # first, read the eigenvalues and compute the explained variances
    # the first line looks like this:
    #  #eigvals: 124.570 78.762    ...
    eigenvalues = f.readline().split()[1:]
    eigenvalues = [float(ev) for ev in eigenvalues]
    explained_variances = [(ev, ev / sum(eigenvalues)) for ev in eigenvalues]
    explained_variances = pd.DataFrame(explained_variances, columns=["eigenvalue", "explained_variance"])

    # next, read the PCs per sample
    df = pd.read_table(f, delimiter=" ", skipinitialspace=True, header=None)
    f.close()
    n_pcs = df.shape[1] - 2
    cols = ["sample_id"] + [f"PC{i}" for i in range(n_pcs)] + ["population"]
    df = df.rename(columns=dict(zip(df.columns,cols)))
    return df, explained_variances


def clean_converted_names(evecs):
    names = []
    for idx, row in evecs.iterrows():
        _, name = row.sample_id.split(":")
        names.append(name)

    evecs.sample_id = names
    return evecs


def annotate_population(empirical, bootstrap):
    populations = []
    for idx, bootstrap_row in bootstrap.iterrows():
        empirical_row = empirical.loc[empirical.sample_id == bootstrap_row.sample_id]
        assert empirical_row.shape[0] == 1, empirical_row.shape[0]

        populations.append(empirical_row.population.item())

    bootstrap["population"] = populations
    return bootstrap


def get_data_for_transformation(empirical, bootstrap, normalize=False, n_pcs=None):
    empirical = empirical.sort_values(by="sample_id").reset_index(drop=True)
    bootstrap = bootstrap.sort_values(by="sample_id").reset_index(drop=True)

    bootstrap_ids = set(bootstrap.sample_id.unique())
    empirical_filtered = empirical.loc[empirical.sample_id.isin(bootstrap_ids)]

    assert bootstrap.shape[0] == empirical_filtered.shape[0]

    if not n_pcs:
        n_pcs = empirical_filtered.shape[1] - 2  # two non-PC columns: sample_id, population
    cols = [f"PC{i}" for i in range(n_pcs)]

    empirical_pcs = empirical_filtered[cols].to_numpy()
    bootstrap_pcs = bootstrap[cols].to_numpy()

    if normalize:
        empirical_pcs = empirical_pcs / np.linalg.norm(empirical_pcs)
        bootstrap_pcs = bootstrap_pcs / np.linalg.norm(bootstrap_pcs)

    return empirical_pcs, bootstrap_pcs