
Working with data from R
========================

Many computational biologists are more familiar in working with R instead of Python and thus might have their own (pre)processing pipeline setup in R.
You can export your data from R as `.RData` file using the ``save`` function and we can then read this data in Python using the ``pyreadr`` package.
This might not work universally for all kinds of data, if you have any issues with exporting your data feel free to contact us and we can work things out together!

First of all, we need to install the ``pyreadr`` package. You can do so either via conda::

    conda install pyreadr -c conda-forge

or via pip::

    pip install pyreadr


Suppose we have the file called ``example.RData``. You can then load this data in python like this:

.. code-block:: python

    import pyreadr
    data = pyreadr.read_r("example.RData")
    print(data.keys())


``data`` is now a Python ``OrderedDict`` and the above code also prints the keys of this dict. Suppose our data in R had the attributes ``geno_data``, ``populations`` and ``sample_ids``, then this will print ``odict_keys(['geno_data', 'sample_ids', 'populations'])``.
We can then access e.g. the ``geno_data`` simply via a dict access and pyreadr will return the data as pandas dataframe. The following code-snipped shows you how you can then transform this data
to a Pandora NumpyDataset.


.. code-block:: python

    import numpy as np
    import pyreadr

    from pandora.dataset import NumpyDataset

    # in our this is an OrderedDict with keys geno_data, populations, and sample_ids
    r_data = pyreadr.read_r("example.RData")

    # these are all pandas dataframes
    geno_data = r_data["geno_data"]
    sample_ids = r_data["sample_ids"]
    populations = r_data["populations"]

    # convert the dataframes to the required input formats for the NumpyDataset
    # geno_data needs to be a numpy NDArray
    geno_data = geno_data.to_numpy()
    # also we need to properly set the nan values in order for the imputation to work
    geno_data = np.nan_to_num(geno_data, nan=np.nan)

    # sample IDs and populations need to be pandas Series, not dataframes
    # in case the sample_ids in R were just a vector, the dataframes will have a column with the same key as above
    # the dataframe will however simply have the
    sample_ids = sample_ids.sample_ids
    populations = populations.populations

    # using this data, we can initialize a NumpyDataset
    dataset = NumpyDataset(geno_data, sample_ids, populations)
