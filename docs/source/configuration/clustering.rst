Clustering Configuration
=====================

The clustering module parameters are specified in a YAML file. A full description of the clustering parameters can be found at `<https://www.proquest.com/openview/23058f048ffb9c4fcb00de75214745aa/1?cbl=18750&diss=y>`_ or `<https://drive.google.com/file/d/1ALDTmtulIdCGkoIVh_DRBip-SKWDeLSK/view?usp=sharing>`_.

The pairwise distance between two features from the feature finding is calculated with the function:

.. math::
    d_{i,j} = (1 - R_{i,j}) + \alpha (1 - e^{-\rho_{i,j} / \tau})

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Variable
     - Definition
   * - :math:`d_{i,j}`
     - Distance between features i and j
   * - :math:`R_{i,j}`
     - Pearson correlation between :math:`\log_{10}(s_i)` and :math:`\log_{10}(s_j)`
   * - :math:`\rho_{i,j}`
     - Root-mean-square deviation between :math:`t_i` and :math:`t_j`
   * - :math:`s_{i,k}`
     - Mean centroid intensity for sample k in feature i
   * - :math:`t_{i,k}`
     - Mean centroid retention time for sample k in feature i
   * - :math:`m_{i,k}`
     - Mean centroid m/z for sample k in feature i




Example Configuration
-------------------

.. code-block:: yaml

    alpha: 0.25
    tau: 4.0
    recursive_clustering: true
    frac_peaks: 0.8
    rt_1sWindow: 5.0
    cluster_outlier_1sWidth: 3.0
    parent_mz_check_intensity_frac: 0.6

Parameters
---------

alpha
    Weighting coefficient for the retention time component of the distance function
tau
    Time constant for the retention time component of the distance function
recursive_clustering
    Whether to perform recursive clustering on large clusters
frac_peaks
    Minimum fraction of peaks to fall within `rt_1sWindow` of a root node required for cluster formation at that node
rt_1sWindow
    1-sided window in seconds for tree children to be considered for clustering with a parent node
cluster_outlier_1sWidth
    1-sided window in seconds for cluster members to be removed as outliers
parent_mz_check_intensity_frac
    Fraction of current cluster parent ion's intensity a child ion with higher m/z must have to be considered a new parent ion
recursive_clustering
    Whether to perform iterative rounds of re-clustering with filtering until the number of clusters is stable (true or false)


