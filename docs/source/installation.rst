Installation
===========


Using Docker (recommended)
-----------

The recommended way to use MetaboZen is through Docker, which ensures all dependencies are correctly installed. You must have Docker installed, for example `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_.

Once Docker is installed, the MetaboZen image can be pulled with:

.. code-block:: bash

    docker pull gkreder/metabozen

And then MetaboZen can be run using:

.. code-block:: bash

   docker run -it -v <DATA_DIRECTORY>:/home/data/ gkreder/metabozen metabozen


Development Installation
----------------------

MetaboZen can also be installed from the Github repo. You must have R and Python installed already. Note that the R dependencies must be installed first. The easiest way to do this may be to use conda in which case all R dependencies can be installed from the following conda packages:
  
  - r-tidyverse == 2.0.0
  - bioconductor-xcms == 4.0.0
  - bioconductor-qvalue == 2.38.0


Or alternatively these R dependencies can be installed from within an R session:

.. code-block:: bash
    install.packages("tidyverse")
    
    if (!require("BiocManager", quietly = TRUE))
        install.packages("BiocManager")

    BiocManager::install("xcms")
    BiocManager::install("qvalue")

Once R dependencies have been installed, MetaboZen and its Python dependencies can be installed:

.. code-block:: bash

   git clone https://github.com/gkreder/metabozen.git
   cd metabozen
   pip install .

Dependencies
-----------

Python Dependencies:
~~~~~~~~~~~~~~~~~~

- Python >= 3.8,<3.13
- numpy >=2.1.2,<3
- pandas >=2.2.3,<3
- scipy >=1.14.1,<2
- scikit-learn >=1.5.2,<2
- matplotlib >=3.9.2,<4
- rpy2 >=3.5.11,<4
- pyteomics >=4.7.5,<5
- tqdm >=4.66.6,<5
- pyyaml >=6.0.2,<7
- xlrd >=2.0.1,<3
- lxml >=5.3.0,<6

R Dependencies:
~~~~~~~~~~~~~

- xcms == 4.0.0
- qvalue == 2.38.0
- tidyverse == 2.0.0
