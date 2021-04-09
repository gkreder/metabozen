################################################################################
# Gabe Reder - gkreder@gmail.com
################################################################################
# A python wrapper for running XCMS on input data (using XCMS 3 syntax)
################################################################################
import sys
import os
import pandas as pd
import numpy as np
import argparse
import hashlib
################################################################################
def peakwidth_format(s):
	try:
		[peak_low, peak_high] = s.replace('(', '').replace(')', '').split(',')
		return((float(peak_low), float(peak_high)))
	except:
		msg = "Not a valid peakwidth format: '{0}'.".format(s)
		raise argparse.ArgumentTypeError(msg)

def prefilter_format(s):
	try:
		[peak_low, peak_high] = s.replace('(', '').replace(')', '').split(',')
		print(peak_low, peak_high)
		return((float(peak_low), float(peak_high)))
	except:
		msg = "Not a valid prefilter format: '{0}'.".format(s)
		raise argparse.ArgumentTypeError(msg)
################################################################################
parser = argparse.ArgumentParser()


subparsers = parser.add_subparsers(dest = 'input_type', required = True)

parser_file = subparsers.add_parser('file')
parser_file.add_argument('--in_file', required = True)
parser_file.add_argument('--debug', action = 'store_true')

parser_args = subparsers.add_parser('args')
################################################
# top-level parameters
################################################
# parser_args.add_argument('--data_dir', required = True)
parser_args.add_argument('--debug', action = 'store_true')
parser_args.add_argument('--in_files', required = True, help = 'Comma separated list of input files', type = str)
parser_args.add_argument('--out_tsv', required = True, type = str)
# parser_args.add_argument('--polarity', required = True, choices = ['positive', 'negative'])
parser_args.add_argument('--sample_groups', required = True, help = 'Comma separated list of sample groups', type = str)
parser_args.add_argument('--sample_names', required = True, help = 'Comma separated list of sample names', type = str)
# parser_args.add_argument('--temp_dir', default = "")
# parser_args.add_argument('--test', action = 'store_true')
# parser_args.add_argument('--use_original_data', action = 'store_true')
# parser_args.add_argument('--keep_R_file', action = 'store_true')
# parser_args.add_argument('--keep_plots', action = 'store_true')
# parser_args.add_argument('--keep_temp_data', action = 'store_true')
# parser_args.add_argument('--file_extension', default='.mzXML')
# parser_args.add_argument('--no_peak_fill', action = 'store_true')
# parser_args.add_argument('--CAMERA', action = 'store_true')
# parser_args.add_argument('--safe_mode', action = 'store_true')
# # integer(1) defining the MS level on which the peak detection should be performed. Defaults to msLevel = 1.
# parser_args.add_argument('--msLevel', default = 1, type = int)



################################################
# centWave parameters
################################################
# maximal tolerated m/z deviation in consecutive scans, in ppm (parts per million)
parser_args.add_argument('--centwave_ppm', type = float, default = 25)
# Chromatographic peak width, given as range (min,max) in seconds
parser_args.add_argument('--centwave_peakwidth', type = peakwidth_format, default=(20.0, 50.0))
# signal to noise ratio cutoff - Signal/Noise ratio, defined as (maxo - baseline)/sd, where maxo is the maximum peak intensity, baseline the estimated baseline value and sd the standard deviation of local chromatographic noise.
parser_args.add_argument('--centwave_snthresh', type = float, default = 10)
# prefilter=c(k,I). Prefilter step for the first phase. Mass traces are only retained if they contain at least k peaks with intensity >= I.
parser_args.add_argument('--centwave_prefilter', type = prefilter_format, default=(3.0,100.0))
# Function to calculate the m/z center of the feature: wMean intensity weighted mean of the feature m/z values, mean mean of the feature m/z values, apex use m/z value at peak apex, wMeanApex3 intensity weighted mean of the m/z value at peak apex and the m/z value left and right of it, meanApex3 mean of the m/z value at peak apex and the m/z value left and right of it.
parser_args.add_argument('--centwave_mzCenterFun', type = str, default = 'wMean')
# Integration method. If =1 peak limits are found through descent on the mexican hat filtered data, if =2 the descent is done on the real data. Method 2 is very accurate but prone to noise, while method 1 is more robust to noise but less exact.
parser_args.add_argument('--centwave_integrate', type = float, default = 1.0)
# minimum difference in m/z for peaks with overlapping retention times, can be negative to allow overlap
parser_args.add_argument('--centwave_mzdiff', type = float, default=-0.001)
# logical, if TRUE a Gaussian is fitted to each peak
parser_args.add_argument('--centwave_fitgauss', choices = ['TRUE', 'FALSE'], default = 'FALSE')
# optional argument which is useful for data that was centroided without any intensity threshold, centroids with intensity < noise are omitted from ROI detection
parser_args.add_argument('--centwave_noise', default = 0, type = float)


# # number of seconds to pause between plotting peak finding cycles
# parser_args.add_argument('--centWave_sleep', default = 0, type = float)
# scan range to process
# parser_args.add_argument('--scanrange', default = 'numeric()')


# parser_args.add_argument('--verbose.columns', choices = ['TRUE', 'FALSE'], default = 'FALSE')
# parser_args.add_argument('--ROI.list', default = 'list()')

################################################
# Grouping (density) parameters
################################################
# Number of grouping/retcor iterations to run 
parser_args.add_argument('--grouping_steps', default = 3, type = int)
# bandwidth (standard deviation or half width at half maximum) of gaussian smoothing kernel to apply to the peak density chromatogram
parser_args.add_argument('--density_bw', default = 30, type = float)
# minimum fraction of samples necessary in at least one of the sample groups for it to be a valid group
parser_args.add_argument('--density_minFraction', default = 0.5, type = float)
# minimum number of samples necessary in at least one of the sample groups for it to be a valid group
parser_args.add_argument('--density_minSamples', default = 1, type = int)
# width of overlapping m/z slices to use for creating peak density chromatograms and grouping peaks across samples
parser_args.add_argument('--density_binSize', default = 0.25, type = float)
# maximum number of groups to identify in a single m/z slice
parser_args.add_argument('--density_maxFeatures', default = 50, type = int)
# # seconds to pause between plotting successive steps of the peak grouping algorithm. peaks are plotted as points showing relative intensity. identified groups are flanked by dotted vertical lines.
# parser_args.add_argument('--grouping_sleep', default = 0, type = float)

########################
# think that mzwid has become binSize in new density function (binsize_density?)
########################
# width of overlapping m/z slices to use for creating peak density chromatograms and grouping peaks across samples
# parser_args.add_argument('--mzwid', default = 0.25, type = float)

################################################
# Retcor (obiwarp) parameters
################################################
# numeric(1) defining the bin size (in mz dimension) to be used for the profile matrix generation. See step parameter in profile-matrix documentation for more details.
parser_args.add_argument('--obiwarp_binSize', default=1, type = float)
# index of sample all others will be aligned to, if NULL sample with most 
# peaks is chosen
# parser_args.add_argument('--obiwarp_centerSample', default='NULL') 
# Responsiveness of warping. 0 will give a linear warp based on start and end points. 100 will use all bijective anchors
parser_args.add_argument('--obiwarp_response', default=1, type = float)
# DistFun function: cor (Pearson's R) or cor_opt (default, calculate only 10% diagonal band of distance matrix, better runtime), cov (covariance), prd (product), euc (Euclidean distance)
parser_args.add_argument('--obiwarp_distFun', default="cor_opt", type = str)
# Penalty for Gap opening - defaults 'cor' = '0.3' 'cov' = '0' 'prd' = '0' 'euc' = '0.9'
# parser_args.add_argument('--gapInit', default='NULL')
# Penalty for Gap enlargement - defaults 'cor' = '2.4' 'cov' = '11.7' 'prd' = 7.8' 'euc' = '1.8'
# parser_args.add_argument('--gapExtend', default='NULL')
# Local weighting applied to diagonal moves in alignment
parser_args.add_argument('--obiwarp_factorDiag', default = 2, type = float)
# Local weighting applied to gap moves in alignment
parser_args.add_argument('--obiwarp_factorGap', default = 1, type = float)
# Local rather than global alignment
parser_args.add_argument('--obiwarp_localAlignment', default = "FALSE", choices=['FALSE', 'TRUE'])
# Penalty for initiating alignment (for local alignment only) Default: 0
parser_args.add_argument('--obiwarp_initPenalty', default = 0, type = float)

################
# Think profStep became binSize param in new obiwarp function
################
# # step size (in m/z) to use for profile generation from the raw data files
# parser_args.add_argument('--profStep', default=1, type = float)
# # vector of colors for plotting each sample
# parser_args.add_argument('--col', default='NULL')
# # vector of line and point types for plotting each sample
# parser_args.add_argument('--ty', default='NULL')


# Required file and parameter inputs 
# parser.add_argument('--in_file', required = True)
args = parser.parse_args()
################################################################################
float_args = ["centwave_ppm", 'centwave_mzdiff', 'centwave_noise', 'obiwarp_factorGap', 'obiwarp_binSize', 'obiwarp_factorDiag', 'obiwarp_response', 'obiwarp_initPenalty']
int_args = ['centwave_integrate', 'grouping_steps']
str_args = ['out_tsv', 'in_files', 'sample_names', 'sample_groups', 'centwave_fitgauss', 'centwave_peakwidth', 'centwave_prefilter', 'centwave_snthresh', 'centwave_mzCenterFun', 'obiwarp_distFun']
if args.input_type == 'file':
	df = pd.read_excel(args.in_file, index_col = 0, names = ['name', 'value'], header = None)
	for (name, row) in df.iterrows():
		val = row['value']
		if name in float_args:
			val = float(val)
		elif name in int_args:
			val = int(val)
		else:
			val = str(val)
		setattr(args, name, val)
	sample_names = args.sample_names
	sample_groups = args.sample_groups
	if sample_names == '':
		sample_names = ','.join([os.path.basename(x) for x in [y.strip() for y in args.in_files.split(',')]])
	else:
		sample_names = ','.join([f'''"{x}"''' for x in sample_names.split(',')])
	if sample_groups == '':
		sample_groups = "integer(length(sample_names))"
	else:
		sample_groups = ','.join([f'''"{x}"''' for x in sample_groups.split(',')])
	# out_dir = os.path.dirname(df.loc['out_tsv'].value)
	# hash_tag = str(int(hashlib.sha256(df.loc['out_tsv'].value.encode('utf-8')).hexdigest(), 16) % 10**12)
	# files = ','.join([f'''"{x}"''' for x in df.loc['in_files'].value.split(',')])
	# sample_names = ','.join([f'''"{x}"''' for x in df.loc['sample_names'].value.split(',')])
	# if str(df.loc['sample_groups']) != "None":
	#     sample_groups = ','.join([f'''"{x}"''' for x in df.loc['sample_groups'].value.split(',')])
	# else:
	#     sample_groups = "integer(length(sample_names))"
else:
	# files = args.in_files
	sample_names = args.sample_names
	sample_names = ','.join([f'''"{x}"''' for x in sample_names.split(',')])
	sample_groups = args.sample_groups
	sample_groups = ','.join([f'''"{x}"''' for x in sample_groups.split(',')])
files = ','.join([f'''"{x}"''' for x in args.in_files.split(',')])

hash_tag = str(int(hashlib.sha256(args.out_tsv.encode('utf-8')).hexdigest(), 16) % 10**12)
out_dir = os.path.dirname(args.out_tsv)
if out_dir == '':
	out_dir = '.'

os.system(f'mkdir -p {out_dir}')
# Create a random suffix for the temporary R file
out_string = ""
# xdata <- adjustRtime(xdata, param=owp)
# xdata <- groupChromPeaks(xdata, param = pdp)
# xdata <- adjustRtime(xdata, param=owp)
# xdata <- groupChromPeaks(xdata, param = pdp)
# xdata <- adjustRtime(xdata, param=owp)
# xdata <- groupChromPeaks(xdata, param = pdp)
grouping_string = "\n".join(["xdata <- adjustRtime(xdata, param=owp)\nxdata <- groupChromPeaks(xdata, param = pdp)" for x in range(args.grouping_steps)])
out_string += f'''library(xcms)
library(plyr)
library(dplyr)
library(stringr)
options(dplyr.summarise.inform = FALSE)
BiocParallel::register(BiocParallel::SerialParam())
files <- c({files})
sample_names <- c({sample_names})
sample_groups <- c({sample_groups})
pd <- data.frame(sample_name = sample_names, sample_group = sample_groups, stringsAsFactors = FALSE)
raw_data <- readMSData(files = files, pdata = new("NAnnotatedDataFrame", pd), mode = "onDisk")
cwp <- CentWaveParam(ppm = {args.centwave_ppm}, mzdiff = {args.centwave_mzdiff}, integrate = {args.centwave_integrate}, fitgauss = {args.centwave_fitgauss}, noise = {args.centwave_noise}, peakwidth=c({args.centwave_peakwidth}), prefilter = c({args.centwave_prefilter}), snthresh = {args.centwave_snthresh}, mzCenterFun = "{args.centwave_mzCenterFun}")
cwp <- CentWaveParam(ppm = {args.centwave_ppm}, mzdiff = {args.centwave_mzdiff}, integrate = {args.centwave_integrate}, fitgauss = {args.centwave_fitgauss}, noise = {args.centwave_noise}, peakwidth=c({args.centwave_peakwidth}), prefilter = c({args.centwave_prefilter}), snthresh = {args.centwave_snthresh}, mzCenterFun = "{args.centwave_mzCenterFun}")
owp <- ObiwarpParam(factorGap = {args.obiwarp_factorGap} , binSize = {args.obiwarp_binSize}, factorDiag = {args.obiwarp_factorDiag} , distFun = "{args.obiwarp_distFun}", response = {args.obiwarp_response} , localAlignment = FALSE, initPenalty = {args.obiwarp_initPenalty} )
pdp <- PeakDensityParam(sampleGroups = xdata$sample_group, minSamples = {args.density_minSamples}, minFraction = {args.density_minFraction}, binSize = {args.density_binSize}, bw = {args.density_bw}, maxFeatures = {args.density_maxFeatures})
xdata <- findChromPeaks(raw_data, param = cwp)
{grouping_string}
xdata <- fillChromPeaks(xdata)
fd <- featureDefinitions(xdata)
fv <- featureValues(xdata, value = 'into')
# fs <- featureSummary(xdata, group = xdata$sample_group)
pd <- phenoData(xdata)
cp <- chromPeaks(xdata)
sample_names <- pd@data$sample_name
peakidxs <- fd$peakidx'''
out_string +='''
suppressWarnings(feature_mzs <- lapply(peakidxs, function (x) {
	if (length(x) == 1){
		cp_x <- cp[x, ]
		sn <- sample_names[cp_x['sample']]
		concat_mz <- data.frame(temp_name = cp_x['mz'])
		names(concat_mz) <- paste('mz', sn, sep = '_')
	}
	else {
		cp_x <- data.frame(cp[x, ])
		sns <- sample_names[cp_x$sample]
		cp_x$sample_name <- sns

		concat_mz <- cp_x %>% dplyr::group_by(sample_name) %>% dplyr::summarise(mz = toString(mz)) # , rt = toString(rt)
		cn_mz <- lapply(concat_mz$sample_name, function (x) { paste('mz', x, sep = '_') })
		concat_mz <- data.frame(t(concat_mz[, -1]))
		colnames(concat_mz) <- cn_mz
	}
	concat_mz
}) %>% plyr::rbind.fill())
feature_mzs$feature_id <- row.names(fv)
feature_mzs <- feature_mzs[, c("feature_id", paste("mz_", sample_names, sep = ""))]

suppressWarnings(feature_rts <- lapply(peakidxs, function (x) {
	if (length(x) == 1){
		cp_x <- cp[x, ]
		sn <- sample_names[cp_x['sample']]
		concat_rt <- data.frame(temp_name = cp_x['rt'])
		names(concat_rt) <- paste('rt', sn, sep = '_')
	}
	else {
		cp_x <- data.frame(cp[x, ])
		sns <- sample_names[cp_x$sample]
		cp_x$sample_name <- sns

		concat_rt <- cp_x %>% dplyr::group_by(sample_name) %>% dplyr::summarise(rt = toString(rt))
		cn_rt <- lapply(concat_rt$sample_name, function (x) { paste('rt', x, sep = '_') })
		concat_rt <- data.frame(t(concat_rt[, -1]))
		colnames(concat_rt) <- cn_rt
	}
	concat_rt
}) %>% plyr::rbind.fill())
feature_rts$feature_id <- row.names(fv)
feature_rts <- feature_rts[, c("feature_id", paste("rt_", sample_names, sep = ""))]

merged_data <- data.frame(merge(fd, fv, by = 'row.names'))
colnames(merged_data)[1] <- "feature_id"
merged_data <- merge(merged_data, feature_rts, by = 'feature_id')
merged_data <- merge(merged_data, feature_mzs, by = 'feature_id')
merged_data <- merged_data %>% dplyr::mutate(name = paste("M", round(mzmed), 'T', round(rtmed), sep="")) %>% select(name, everything())

merged_data$peakidx <- str_replace_all(merged_data$peakidx, "[\\r\\n]" , "")

names <- as.character(merged_data$name)
merged_data$name <- ifelse(duplicated(names) | duplicated(names, fromLast=TRUE), paste(names, ave(names, names, FUN=seq_along), sep='_'), names)'''

out_string += f'''
write.table(merged_data, '{args.out_tsv}', sep='\\t', col.names=NA)
cpData <- chromPeaks(xdata)
cpData <- write.table(cpData, '{args.out_tsv.replace('.tsv', '_CPDATA.tsv')}', sep='\\t', col.names=NA)
saveRDS(xdata, file = '{args.out_tsv.replace('.tsv', '_XCMSSET.rds')}')
'''

r_fname = os.path.join(out_dir, f"xcms_{hash_tag}.R")
params_fname = os.path.join(out_dir, f"params_{hash_tag}.log")
with open(params_fname, 'w') as f:
	print(' '.join(sys.argv), file = f)
	print('', file = f)
	print(args, file = f)
with open(r_fname, 'w') as f:
    print(out_string, file = f)
os.system(f"Rscript {r_fname}")
if not args.debug:
	os.system(f"rm {r_fname}")

