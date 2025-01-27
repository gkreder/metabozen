Usage Guide
=============

MetaboZen provides two ways to run analyses:

Individual Steps
-------------------

Run each step separately:

.. code-block:: bash

   metabozen feature-finding -p params.yaml -s samples.xlsx -o features.tsv
   metabozen clustering -i features.tsv -p params.yaml -s samples.xlsx -o clusters.tsv
   metabozen stats-tests -i clusters.tsv -p params.yaml -s samples.xlsx -o stats.tsv

Complete Pipeline
-------------------

Run all steps with a single configuration file:

.. code-block:: bash

   metabozen pipeline pipeline_params.yaml

Configuration Files
----------------------

MetaboZen uses YAML configuration files for each step. See the :doc:`configuration/index` section for detailed parameter descriptions.

Sample Information File
-------------------------

The samples file (Excel, CSV, or TSV) should contain the following columns:

- ``filename``: Path to the input file
- ``sample_name``: Unique identifier for the sample
- ``sample_group``: Group identifier for statistical analysis
- ``normalization`` (optional): Normalization factor

For detailed API documentation, see the :doc:`api/index` section.
