Feature Finding
=================

.. automodule:: metabozen.feature_finding
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
-------------------------

The feature finding module can be run from the command line using:

.. code-block:: bash

   metabozen-feature-finding -p params.yaml -s samples.xlsx -o features.tsv

Arguments
~~~~~~~~~~~~

--parameters, -p
    YAML configuration file with XCMS parameters
--samples, -s
    Excel/CSV/TSV file containing sample information
--out_file, -o
    Path to output TSV/CSV file
--no_logfile, -n
    Disable saving log to a file
--debug_files, -d
    Save output .rds and chromPeak files for debugging
--ipdb_debug
    Enable debugging with ipdb 