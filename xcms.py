import pandas as pd
import argparse
import logging
from pathlib import Path
import hashlib
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import yaml

# Enable the conversion between pandas and R data frames
pandas2ri.activate()

# Import necessary R packages
base = importr('base')
utils = importr('utils')
msnbase = importr('MSnbase')
xcms = importr('xcms')
dplyr = importr('dplyr')
stringr = importr('stringr')
BiocParallel = importr('BiocParallel')
stats = importr('stats')
methods = importr('methods')
biobase = importr('Biobase')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run XCMS on input data")
    parser.add_argument('--config', required=True, help='YAML configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (keep temporary files)')
    return parser.parse_args()

################################################################################
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = {k: v for k, v in config.items() if v != None}
    return config

################################################################################
def create_output_directory(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

################################################################################
def run_xcms(config):
    # Extract configuration values
    in_files = ro.StrVector(config['in_files'].split(','))
    sample_names = ro.StrVector(config['sample_names'].split(','))
    sample_groups = ro.StrVector(config['sample_groups'].split(',')) if config['sample_groups'] != "None" else stats.integer(len(sample_names))

    # Create data frame for sample data
    sdf = ro.DataFrame({'sample_name': sample_names, 'sample_group': sample_groups})

    BiocParallel.register(BiocParallel.SerialParam())

    raw_data = msnbase.readMSData(files=in_files, pdata=methods.new('AnnotatedDataFrame', sdf), mode='onDisk')
    cwp = xcms.CentWaveParam(
        ppm=config.get('centwave_ppm', 25),
        mzdiff=config.get('centwave_mzdiff', 0.01),
        integrate=config.get('centwave_integrate', 1),
        fitgauss=config.get('centwave_fitgauss', False),
        noise=config.get('centwave_noise', 1000),
        peakwidth=ro.FloatVector([float(x) for x in config['centwave_peakwidth'].split(',')]),
        prefilter=ro.IntVector([float(x) for x in config['centwave_prefilter'].split(',')]),
        snthresh=config.get('centwave_snthresh', 10),
        mzCenterFun=config.get('centwave_mzCenterFun', 'wMean')
    )

    xdata = xcms.findChromPeaks(raw_data, param=cwp)

    owp = xcms.ObiwarpParam(
        factorGap=float(config.get('obiwarp_factorGap', 0.5)),
        binSize=float(config.get('obiwarp_binSize', 0.1)),
        factorDiag=float(config.get('obiwarp_factorDiag', 2)),
        distFun=config.get('obiwarp_distFun', 'cor'),
        response=float(config.get('obiwarp_response', 1)),
        localAlignment=False,
        initPenalty=float(config.get('obiwarp_initPenalty', 0.05))
    )
    pheno_data = xdata.slots['phenoData']
    pheno_data_df = pheno_data.slots['data']
    sample_groups = pheno_data_df['sample_group']
    sample_groups_r = ro.StrVector(sample_groups)
    pdp = xcms.PeakDensityParam(
        sampleGroups=sample_groups_r,
        minSamples=config.get('density_minSamples', 1),
        minFraction=config.get('density_minFraction', 0.5),
        binSize=config.get('density_binSize', 0.005),
        bw=config.get('density_bw', 5),
        maxFeatures=config.get('density_maxFeatures', 50)
    )

    grouping_steps = config.get('grouping_steps', 3)
    for gs in range(grouping_steps):
        xdata = xcms.adjustRtime(xdata, param=owp)
        xdata = xcms.groupChromPeaks(xdata, param=pdp)
    xdata = xcms.fillChromPeaks(xdata)

    fd = xcms.featureDefinitions(xdata)
    fv = xcms.featureValues(xdata, value='into')

    sdf = xcms.phenoData(xdata)
    cp = xcms.chromPeaks(xdata)
    sample_names = sdf.slots['data']['sample_name']

    list_data = fd.do_slot('listData')
    df_fd = pd.DataFrame(dict(zip(list_data.names, map(list,list(list_data)))))
    df_fd['peakidx'] = df_fd['peakidx'].apply(lambda x : np.array(x).astype(int))
    df_fd.insert(0, "feature_id", fd.slots['rownames'])
    peakidxs = df_fd['peakidx'].values
    df_fv = pd.DataFrame(fv, columns = df_pd.sample_name)
    df_pd = pheno_data.slots['data']
    df_pd.index = df_pd.index.astype(int)
    df_cp = pd.DataFrame(cp, columns = ["mz","mzmin","mzmax","rt","rtmin","rtmax","into","intb","maxo","sn","sample"])
    df_cp['sample'] = df_cp['sample'].astype(int)
    # fd_test = xdata.slots['featureData']
    # df_fd_test = fd_test.slots['data'] # this DF contains spectrum information for each spectrum
    result = []
    peak_description_rows = []
    for x in peakidxs:
        df_cp_x = df_cp.iloc[x - 1].copy()
        sns = df_pd.loc[df_cp_x['sample'].values]['sample_name'].values
        df_cp_x['sample_name'] = sns
        df_cp_x_summary = df_cp_x.groupby('sample_name').mean()[['mz', 'rt']]
        row = {f"rt_{k}" : v for k,v in df_cp_x_summary['rt'].to_dict().items()}
        row.update({f"mz_{k}" : v for k,v in df_cp_x_summary['mz'].to_dict().items()})
        peak_description_rows.append(row)
    df_peak_description = pd.DataFrame(peak_description_rows)
    for sn in df_pd['sample_name']:
        if sn not in df_peak_description.columns:
            df_peak_description[sn] = None
    def sort_fun(x):
        sort_val = list(df_pd['sample_name'].values).index(x.replace('rt_', '').replace('mz_', ''))
        if 'mz_' in x:
            sort_val += len(df_pd) + 1
        return sort_val
    df_peak_description = df_peak_description[sorted(df_peak_description.columns, key = lambda x : sort_fun(x))]
    df_merged = pd.concat([df_fd, df_fv, df_peak_description], axis = 1)
    df_merged.insert(0, "name", df_merged.apply(lambda row: f"M{round(row['mzmed'])}T{round(row['rtmed'])}", axis=1))

    out_tsv = config['out_tsv']
    if out_tsv.endswith('.tsv'):
        out_sep = '\t'
    elif out_tsv.endswith('.csv'):
        out_sep = ','
    else:
        raise ValueError(f"Output file must be either a .tsv or .csv file: {out_tsv}")
    df_merged.to_csv(out_tsv, sep=out_sep, index=False)
    df_cp.to_csv(out_tsv.replace('.tsv', '_CPDATA.tsv'), sep='\t', index=False)

    # cp_data = pd.DataFrame(xcms.chromPeaks(xdata))
    # cp_data.to_csv(out_tsv.replace('.tsv', '_CPDATA.tsv'), sep='\t', index=False)

    ro.r['saveRDS'](xdata, file=out_tsv.replace('.tsv', '_XCMSSET.rds'))

################################################################################
def main():
    args = parse_arguments()
    config = read_config(args.config)
    out_dir = Path(config['out_tsv']).parent
    create_output_directory(out_dir)
    run_xcms(config)

if __name__ == "__main__":
    main()
