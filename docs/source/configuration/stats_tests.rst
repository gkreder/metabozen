Statistical Tests Configuration
===========================

The statistical tests module parameters are specified in a YAML file.

Example Configuration
-------------------

.. code-block:: yaml

   # Statistical test parameters
   mann_whitney_u:
     group_1: "control"
     group_2: "treatment"
     kwargs:
       paired: false
       qvalues: true
   
   fold_change:
     method: "mean"
     log2: true
   
   visualization:
     volcano_plot: true
     pca_plot: true
     heatmap: true

Parameters
---------

mann_whitney_u
    Configuration for Mann-Whitney U test
    
    group_1
        Name of first group in samples file
    group_2
        Name of second group in samples file
    kwargs
        Additional arguments for statistical test
        
        paired
            Whether to perform paired test
        qvalues
            Whether to calculate q-values

fold_change
    Configuration for fold change calculation
    
    method
        Method for calculating fold change ("mean" or "median")
    log2
        Whether to calculate log2 fold change

visualization
    Configuration for visualization options
    
    volcano_plot
        Generate volcano plot
    pca_plot
        Generate PCA plot
    heatmap
        Generate heatmap 