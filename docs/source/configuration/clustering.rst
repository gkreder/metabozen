Clustering Configuration
=====================

The clustering module parameters are specified in a YAML file.

Example Configuration
-------------------

.. code-block:: yaml

   # Clustering parameters
   alpha: 0.25          # Similarity threshold
   tau: 4.0            # Time tolerance in seconds
   frac_peaks: 0.8     # Minimum fraction of peaks required
   recursive_clustering: true  # Enable recursive clustering

   # Optional parameters
   min_cluster_size: 3
   max_cluster_size: 100
   rt_tolerance: 30.0  # Retention time tolerance in seconds

Parameters
---------

alpha
    Similarity threshold for clustering (0.0 to 1.0)
tau
    Time tolerance in seconds for peak alignment
frac_peaks
    Minimum fraction of peaks required for cluster formation
recursive_clustering
    Whether to perform recursive clustering on large clusters
min_cluster_size
    Minimum number of features in a cluster (optional)
max_cluster_size
    Maximum number of features in a cluster (optional)
rt_tolerance
    Retention time tolerance in seconds (optional) 