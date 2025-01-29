Clustering
============

.. automodule:: metabozen.clustering
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
-------------------------

The clustering module can be run from the command line using:

.. code-block:: bash

   metabozen-clustering -i features.tsv -p params.yaml -s samples.xlsx -o clusters.tsv

Arguments
~~~~~~~~~~~~

--clustering_in_file, -i
    TSV/CSV file produced from XCMS feature finding run
--parameters, -p
    YAML configuration file with clustering parameters
--samples, -s
    Excel/CSV/TSV file containing sample information
--out_file, -o
    Path to output TSV/CSV file
--no_logfile, -n
    Disable saving log to a file
--ipdb_debug
    Enable debugging with ipdb 