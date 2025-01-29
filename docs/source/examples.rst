Examples
========

Complete Workflow Example
---------------------------

This example demonstrates a complete metabolomics analysis workflow assuming you have prepared some mzML files for analysis.

1. Prepare Your Sample Sheet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a samples file (``samples.tsv``):

.. code-block:: text

   Sample Name    Sample Group    File                      Normalization
   05_IC05        IC              /home/data/05_IC05.mzML   0.035631082
   01_IC01        IC              /home/data/01_IC01.mzML   0.262732852
   18_CL01        CL              /home/data/18_CL01.mzML   0.21041133
   19_CL02        CL              /home/data/19_CL02.mzML   0.051267368


2. Feature Finding
~~~~~~~~~~~~~~~~~~~~~

Create ``feature_finding_params.yaml``:

.. code-block:: yaml

   # CentWave parameters
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
   # Obiwarp parameters
   obiwarp:
     factorGap: 1.0
     binSize: 0.5
     factorDiag: 2.0
     distFun: "cor_opt"
     response: 1.0
     localAlignment: false
     initPenalty: 0.0
   # PeakDensity parameters
   density:
     minSamples: 1
     minFraction: 0.25
     binSize: 0.025
     bw: 3.0
     maxFeatures: 200
   grouping_steps: 3

Run feature finding:

.. code-block:: bash

   metabozen feature-finding -p feature_finding_params.yaml -s samples.tsv -o features.tsv

3. Clustering
~~~~~~~~~~~~~~~~

Create ``clustering_params.yaml``:

.. code-block:: yaml

   alpha: 0.25
   tau: 4.0
   frac_peaks: 0.8
   recursive_clustering: true
   rt_1sWindow: 5.0
   cluster_outlier_1sWidth: 3.0
   parent_mz_check_intensity_frac: 0.6

Run clustering:

.. code-block:: bash

   metabozen clustering -i features.tsv -p clustering_params.yaml -s samples.tsv -o clusters.tsv

4. Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``stats_params.yaml``:

.. code-block:: yaml

   mann_whitney_u:
     group_1: "IC"
     group_2: "CL"
     kwargs:
       paired: false
       qvalues: true

Run statistical analysis:

.. code-block:: bash

   metabozen stats-tests -i clusters.tsv -p stats_params.yaml -s samples.tsv -o stats.tsv

5. Complete Pipeline
~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can run the complete pipeline with a single configuration file. Create ``pipeline_params.yaml``:

.. code-block:: yaml

   samples: "samples.tsv"
   out_dir: "results/"
   run_name: "analysis"
   feature_finding:
     parameters: "feature_finding_params.yaml"
     out_file: "features.tsv"
     debug_files: true
   clustering:
     parameters: "clustering_params.yaml"
     out_file: "clusters.tsv"
     debug_files: true
   stats_tests:
     parameters: "stats_params.yaml"
     out_file: "stats.tsv"
     plot_format: "svg"

Run the complete pipeline:

.. code-block:: bash

   metabozen pipeline pipeline_params.yaml
