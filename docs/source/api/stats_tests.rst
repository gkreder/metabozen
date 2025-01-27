Statistical Tests
===================

.. automodule:: metabozen.stats_tests
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
--------------------------

The statistical tests module can be run from the command line using:

.. code-block:: bash

   metabozen-stats-tests -i clusters.tsv -p params.yaml -s samples.xlsx -o stats.tsv

Arguments
~~~~~~~~~~~~~

--clustering_in_file, -i
    TSV/CSV file from clustering output
--parameters, -p
    YAML configuration file with statistical test parameters
--samples, -s
    Excel/CSV/TSV file containing sample information
--out_file, -o
    Path to output TSV/CSV file
--no_logfile, -n
    Disable saving log to a file
--no_plots, -np
    Disable saving plots
--plot_format
    Format for saving plots (png, svg, jpg, jpeg)
--ipdb_debug
    Enable debugging with ipdb 