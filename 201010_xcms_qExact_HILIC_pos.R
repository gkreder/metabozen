library(xcms)
library(plyr)
library(dplyr)
library(stringr)
BiocParallel::register(BiocParallel::SerialParam())

##############################################################################################################################
# MAKE SURE THESE ARE ORDERED
##############################################################################################################################
files <- c("/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL1_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL2_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL3_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL5_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL6_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL7_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL8_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL9_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL10_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL11_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL14_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/CL16_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC1_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC2_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC3_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC4_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC5_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC6_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC7_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC8_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC9_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC10_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC11_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC12_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC13_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC14_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC15_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC16_pos_HILIC.mzML",
"/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/unzipped_data/200630_Meyer_Q_Exactive_Files/raw_data_files/HILIC/Pos/IC17_pos_HILIC.mzML")

sample_names <- c("CL1",
"CL2",
"CL3",
"CL5",
"CL6",
"CL7",
"CL8",
"CL9",
"CL10",
"CL11",
"CL14",
"CL16",
"IC1",
"IC2",
"IC3",
"IC4",
"IC5",
"IC6",
"IC7",
"IC8",
"IC9",
"IC10",
"IC11",
"IC12",
"IC13",
"IC14",
"IC15",
"IC16",
"IC17")
sample_groups <- c("CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"CL",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC",
"IC")
##############################################################################################################################
pd <- data.frame(sample_name = sample_names, sample_group = sample_groups, stringsAsFactors = FALSE)
raw_data <- readMSData(files = files, pdata = new("NAnnotatedDataFrame", pd), mode = "onDisk")
cwp <- CentWaveParam(ppm = 30, mzdiff = 0.01, integrate = 1.0, fitgauss = FALSE, noise = 0, peakwidth=c(10.0,60.0), prefilter = c(3.0, 500.0), snthresh = 10, mzCenterFun = "wMean")
xdata <- findChromPeaks(raw_data, param = cwp)
owp <- ObiwarpParam(factorGap = 1, binSize = 0.5, factorDiag = 2, distFun = "cor_opt", response = 1, localAlignment = FALSE, initPenalty = 0)
pdp <- PeakDensityParam(sampleGroups = xdata$sample_group, minSamples = 1, minFraction = 0.25, binSize = 0.025, bw = 3, maxFeatures = 200)
xdata <- adjustRtime(xdata, param=owp)
xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- adjustRtime(xdata, param=owp)
xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- adjustRtime(xdata, param=owp)
xdata <- groupChromPeaks(xdata, param = pdp)
xdata <- fillChromPeaks(xdata)
fd <- featureDefinitions(xdata)
fv <- featureValues(xdata, value = 'into')
# fs <- featureSummary(xdata, group = xdata$sample_group)
pd <- phenoData(xdata)
cp <- chromPeaks(xdata)
sample_names <- pd@data$sample_name
peakidxs <- fd$peakidx

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

merged_data$peakidx <- str_replace_all(merged_data$peakidx, "[\r\n]" , "")

names <- as.character(merged_data$name)
merged_data$name <- ifelse(duplicated(names) | duplicated(names, fromLast=TRUE), paste(names, ave(names, names, FUN=seq_along), sep='_'), names)

write.table(merged_data, '/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/xcms_runs/HILIC_pos/XCMSnEXP.tsv', sep='\t', col.names=NA)
cpData <- chromPeaks(xdata)
cpData <- write.table(cpData, '/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/xcms_runs/HILIC_pos/CpDATA.tsv', sep='\t', col.names=NA)
saveRDS(xdata, file = '/media/gkreder/5TB/data/mass_spec/peak_clustering/200630_qExactive_runs/xcms_runs/HILIC_pos/xcmsSet.rds')
