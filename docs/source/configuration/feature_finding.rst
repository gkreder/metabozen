Feature Finding Configuration
=================================

Feature finding configuration parameters are passed to a run through an input YAML file. 

The feature finding module is a wrapper to `XCMS <https://bioconductor.org/packages/release/bioc/html/xcms.html>`_ 

Parameters are grouped by XCMS function (CentWave, Obiwarp, and PeakDensity) respectively corresponding to within sample peak detection, peak retention time (RT) correction, and inter-sample peak group finding.

The `grouping_steps` parameter refers to the number of RT correction + grouping iterations to perform.

Example Configuration
------------------------

.. code-block:: yaml

    # CentWave parameters
    centwave:
        ppm: 30.0
        mzdiff: 0.01
        integrate: 1
        fitgauss: false
        noise: 0.0
        peakwidth: [10.0,60.0]
        prefilter: [3.0,500.0]
        snthresh: 10.0
        mzCenterFun: "wMean"
    # Obiwarp parameters (adjustRtime)
    obiwarp:
        factorGap: 1.0
        binSize: 0.5
        factorDiag: 2.0
        distFun: "cor_opt"
        response: 1.0
        localAlignment: false
        initPenalty: 0.0
    # PeakDensity parameters (do_groupChromPeaks_density)
    density:
        minSamples: 1
        minFraction: 0.25
        binSize: 0.025
        bw: 3.0
        maxFeatures: 200
    grouping_steps: 3

Parameters
-------------

CentWave
^^^^^^^^^^^
More detailed explanations of the centwave parameters can be found at 
`<https://tkimhofer.github.io/msbrowser/articles/pars.html#mzcenterfun-mz-summary-statistic-of-a-peak>`_

The original centwave paper can be found at `<https://link.springer.com/article/10.1186/1471-2105-9-504>`_

ppm
    Allowed signal deviation in m/z dimension
peakwidth
    Range of peak elution times (in seconds) [min, max]
snthresh
    Threshold signal to noise ratio
prefilter
    Peak definition: Number of data points (n) exceeding a certain intensity threshold (I) [n, I]
mzdiff
    Accepted closeness of two signals in m/z dimension
integrate
    Integration method (1 = sum of intensities, 2 = peak area)
fitgauss
    Whether to perform peak parameterisation using Gaussian distribution (false or true)
noise
    Intensity cut-off, values below are considered as instrument noise
mzCenterFun
    Function to calculate the m/z center of a chromatographic peak ("apex", "mean", "wMean", "meanApex3", or "wMeanApex3")


Obiwarp
^^^^^^^^^^^
More detailed explanations of the obiwarp process and some parameters can be found at `<https://hackmd.io/@preprocess/r1lgodGz_>`_. 

Documentation for the XCMS `adjustRtime` function can be found at `<https://sneumann.github.io/xcms/reference/adjustRtime.html>`_.

The original obiwarp paper can be found at `<https://pubmed.ncbi.nlm.nih.gov/16944896/>`_.

factorGap
    Local weighting applied to gap moves in alignment
binSize
    Bin size (in mz dimension) to be used for the profile matrix generation.
factorDiag
    The local weight applied to diagonal moves in the obiwarp alignment
distFun
    Distance function to be used in obiwarp. Allowed values are "cor" (Pearson's correlation), "cor_opt" (calculate only 10% diagonal band of distance matrix; better runtime), "cov" (covariance), "prd" (product) and "euc" (Euclidian distance)
response
    The responsiveness of warping with response = 0 giving linear warping on start and end points and response = 100 warping using all bijective anchors.
localAlignment
    Whether a local alignment should be performed instead of the default global alignment (false or true)
initPenalty
    The penalty for initiating an alignment (for local alignment only)
    

PeakDensity
^^^^^^^^^^^^

PeakDensity grouping of peaks across samples is described in the original XCMS publication at `<https://pubs.acs.org/doi/full/10.1021/ac051437y>`_.

Documentation for the XCMS `do_groupChromPeaks_density` function can be found at `<https://sneumann.github.io/xcms/reference/do_groupChromPeaks_density.html>`_.

minSamples
    The minimum number of samples in at least one sample group in which the peaks have to be detected to be considered a peak group (feature)
minFraction
    The minimum fraction of samples in at least one sample group in which the peaks have to be present to be considered as a peak group (feature)
binSize
    The size of the overlapping slices in m/z dimension
bw
    The bandwidth (standard deviation ot the smoothing kernel) to be used
maxFeatures
    The maximum number of peak groups to be identified in a single mz slice





