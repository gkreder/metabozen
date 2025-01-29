Statistical Tests Configuration
==================================

The statistical tests module parameters are specified in a YAML file. Currently, only the Mann-Whitney U test is supported.

Example Configuration
--------------------------

.. code-block:: yaml

   # Statistical test parameters
   mann_whitney_u:
    group_1: "IC"
    group_2: "CL"
    kwargs: 
        paired: false
        qvalues: true

Parameters
-----------

mann_whitney_u
    Configuration for Mann-Whitney U test

    group_1
        Name of first sample group in the input data (e.g. "IC")
    group_2
        Name of second sample group in the input data (e.g. "CL")
    kwargs
        Additional arguments for statistical test
        
        paired
            Whether to perform paired test (true or false)
        qvalues
            Whether to calculate q-values (true or false)

