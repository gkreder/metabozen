Examples
========

Complete Workflow Example
-----------------------

This example demonstrates a complete metabolomics analysis workflow:

1. Prepare Your Data
~~~~~~~~~~~~~~~~~~

Create a samples file (``samples.xlsx``):

.. code-block:: text

   filename        sample_name    sample_group    normalization
   data/s1.mzML    sample1        control         1.0
   data/s2.mzML    sample2        control         1.0
   data/s3.mzML    sample3        treatment       1.0
   data/s4.mzML    sample4        treatment       1.0

2. Feature Finding
~~~~~~~~~~~~~~~~

Create ``xcms_params.yaml``:

.. code-block:: yaml

   centwave:
     ppm: 30.0
     peakwidth: [10.0, 60.0]
     snthresh: 10.0
     prefilter: [3, 500]
     mzdiff: 0.01
     integrate: 1
     fitgauss: false
     noise: 0.0
     mzCenterFun: "wMean"

Run feature finding:

.. code-block:: bash

   metabozen-feature-finding -p xcms_params.yaml -s samples.xlsx -o features.tsv

3. Clustering
~~~~~~~~~~~

Create ``clustering_params.yaml``:

.. code-block:: yaml

   alpha: 0.25
   tau: 4.0
   frac_peaks: 0.8
   recursive_clustering: true

Run clustering:

.. code-block:: bash

   metabozen-clustering -i features.tsv -p clustering_params.yaml -s samples.xlsx -o clusters.tsv

4. Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

Create ``stats_params.yaml``:

.. code-block:: yaml

   mann_whitney_u:
     group_1: "control"
     group_2: "treatment"
     kwargs:
       paired: false
       qvalues: true

Run statistical analysis:

.. code-block:: bash

   metabozen-stats-tests -i clusters.tsv -p stats_params.yaml -s samples.xlsx -o stats.tsv
