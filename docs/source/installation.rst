Installation
===========

There are several ways to install and use MetaboZen:

Using pip
---------

.. code-block:: bash

   pip install metabozen

Using conda
----------

.. code-block:: bash

   conda install -c gkreder metabozen

Using Docker
-----------

The recommended way to use MetaboZen is through Docker, which ensures all dependencies are correctly installed:

.. code-block:: bash

   docker run -it -v <DATA_DIRECTORY>:/home/data/ <SCRIPTS_DIRECTORY>:/home/scripts gkreder/py_metab bash

Development Installation
----------------------

To install MetaboZen for development:

.. code-block:: bash

   git clone https://github.com/gkreder/metabozen.git
   cd metabozen
   pip install -e .

Dependencies
-----------

Python Dependencies:
~~~~~~~~~~~~~~~~~~

- Python >= 3.8
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- rpy2
- pyteomics
- tqdm
- pyyaml
- xlrd
- lxml

R Dependencies:
~~~~~~~~~~~~~

- xcms
- qvalue
- tidyverse
