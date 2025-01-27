Feature Finding Configuration
==========================

The feature finding module uses XCMS parameters specified in a YAML file.

Example Configuration
-------------------

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

Parameters
---------

ppm
    Mass accuracy in parts per million
peakwidth
    Expected peak width range in seconds [min, max]
snthresh
    Signal-to-noise ratio cutoff
prefilter
    Prefilter step [k, I]. Peaks with k scans and I intensity are kept
mzdiff
    Minimum difference in m/z for peaks with overlapping retention times
integrate
    Integration method (1 = sum of intensities, 2 = peak area)
fitgauss
    Whether to fit Gaussian peaks
noise
    Noise level
mzCenterFun
    Method to determine peak m/z ("wMean" or "mean") 