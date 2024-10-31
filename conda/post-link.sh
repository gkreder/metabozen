#!/bin/bash
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("xcms")
BiocManager::install("qvalue")




