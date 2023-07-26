Installation
============

Requirements
------------
Pandora makes use of the `Eigensoft <https://github.com/DReichLab/EIG>`_ software package, more specifically the ``smartpca`` and ``convertf`` tools.
Before using Pandora, make sure you have smartpca and convertf installed. The easiest way to install Eigensoft is to use conda:

.. code-block:: bash

    conda install eigensoft -c bioconda

If you don't want to use conda, you can also follow the instructions in the linked Eigensoft repo.

Note that both options will not work if you are working on a MacBook with an M1 or M2 chip! If this is the case,
follow the instructions below to get Eigensoft running on your laptop. Note, that we recommend running Pandora on a
larger machine than your laptop since the implemented bootstrap procedure runs multiple PCA/MDS analyses and will likely
take a while and might not even be feasible on a machine with only little RAM.

Installing Pandora
------------------
Install using conda (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Coming soon, stay tuned :-)

Install using pip
^^^^^^^^^^^^^^^^^
You can also install Pandora using the python package manager pip:

.. code-block:: bash

    pip install git+https://github.com/tschuelia/Pandora.git

Please note that this can lead to issues with package versions and dependencies when installing in an existing (conda) environment.
Verify the correct installation by running ``pandora -h``.


Troubleshooting
---------------

Installing Eigensoft on MacBooks with M1/M2 chips
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First of all, you should make sure that you are actually working on a MacBook with M1/M2 chip.
You can do so by typing ``arch`` in your command line tool. If you see ``arm64`` continue with the instructions below.
Otherwise refer to the installation instructions above. Please make absolutely sure that you are on an M1/M2 Mac and
you have no other option than to work on your Mac, because the following installation is a terrible hack.
It works, but it's ugly and every software engineer would (rightfully) get angry with me for even suggesting this solution ;-)

To install Eigensoft we will make use of ``micromamba`` and its platform tag. You will need conda for this.

1. Install ``micromamba``

.. code-block:: bash

    conda install micromamba

2. Create a new micromamba environment

.. code-block:: bash

    micromamba env create -n eigensoft_env

3. Install Eigensoft

.. code-block:: bash

    micromamba install -n eigensoft_env eigensoft --platform osx-64

4. Next, we need to find the directory where eigensoft was installed to. To do so, type

.. code-block:: bash

    micromamba list -n eigensoft_env

5. In the output you should see a line similar to this:

.. code-block:: bash

    List of packages in environment: "/Users/julia/micromamba/envs/eigensoft_env"

6. Using the path that is stated there, we can manually concat the exec paths for ``smartpca`` and ``convertf``
by appending ``/bin/smartpca`` and ``/bin/convertf``. So the full paths for ``smarptca`` and ``convertf`` will be something like

.. code-block:: bash

    /Users/julia/micromamba/envs/eigensoft_env/bin/smartpca
    /Users/julia/micromamba/envs/eigensoft_env/bin/convertf

7. Verify that the path is correct by typing them in your terminal one after the other. For ``convertf`` you should see
an output with ``fatalx:`` and for ``smartpca`` something with ``no parameters``.
If the output says something about ``unknown command`` please open an issue and I'm sure we can figure out what is going wrong.

8. In the Pandora config file (see Usage for more details) make sure to set the options ``smartpca`` and ``convertf``
to the respective paths. So your config should then contain the following:

.. code-block:: yaml

    smartpca: /Users/julia/micromamba/envs/eigensoft_env/bin/smartpca
    convertf: /Users/julia/micromamba/envs/eigensoft_env/bin/convertf
