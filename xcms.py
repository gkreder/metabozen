import pandas as pd
import argparse
import logging
from pathlib import Path
import hashlib
import yaml
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run XCMS on input data")
    parser.add_argument('--parameters', '-p', required=True, help='YAML configuration file with parameters')
    parser.add_argument('--samples', '-s', required=True, help='tsv/csv/xlsx file containing input sample information')
    parser.add_argument('--output', '-o', required=True, help='Path to output TSV file (parent directory will be created if nonexistent)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode (keep temporary files)')
    return parser.parse_args()

def load_R_libraries():
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
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

################################################################################
def validate_centwave(config):
    for cwp in ['peakwidth', 'prefilter']:
        if cwp in config['centwave']:
            cwp_val = config['centwave'][cwp]
            assert type(cwp_val) == list and len(cwp_val) == 2 and np.all([type(x) == float for x in cwp_val]), f"centwave.{cwp} must be a list of two float values: {cwp_val}"
    for cwp in ['ppm', 'mzdiff', 'snthresh', 'noise']:
        if cwp in config['centwave']:
            cwp_val = config['centwave'][cwp]
            assert type(cwp_val) in [float, int], f"centwave.{cwp} must be a float: {cwp_val}"
            config['centwave'][cwp] = float(cwp_val)
    if 'integrate' in config['centwave']:
        cwp = config['centwave']['integrate']
        assert type(cwp) == int and ( cwp == 0 or cwp == 1), f"centwave.integrate must be an integer with value 0 or 1: {cwp}"
    if 'fitgauss' in config['centwave']:
        cwp = config['centwave']['fitgauss']
        assert type(cwp) == bool, f"centwave.fitgauss must be a boolean: {cwp}"
    if 'mzCenterFun' in config['centwave']:
        cwp = config['centwave']['mzCenterFun']
        allowed_vals = ['wMean', 'mean', 'apex', 'wMeanApex3','meanApex3']
        assert type(cwp) == str and cwp in allowed_vals, f"centwave.mzCenterFun must be one of {allowed_vals}: {cwp}"
    return config

################################################################################
def validate_obiwarp(config):
    for cwp in ['factorGap', 'binSize', 'factorDiag', 'response', 'initPenalty']:
        if cwp in config['obiwarp']:
            cwp_val = config['obiwarp'][cwp]
            assert type(cwp_val) in [float, int], f"obiwarp.{cwp} must be a float: {cwp_val}"
            config['obiwarp'][cwp] = float(cwp_val)
    if 'localAlignment' in config['obiwarp']:
        cwp = config['obiwarp']['localAlignment']
        assert type(cwp) == bool, f"obiwarp.localAlignment must be a boolean: {cwp}"
    if 'distFun' in config['obiwarp']:
        cwp = config['obiwarp']['distFun']
        allowed_vals = ['cor', 'cor_opt', 'cov', 'prd', 'euc']
        assert type(cwp) == str and cwp in allowed_vals, f"obiwarp.distFun must be one of {allowed_vals}: {cwp}"
    return config

################################################################################
def validate_density(config):
    for cwp in ['minSamples', 'maxFeatures']:
        if cwp in config['density']:
            cwp_val = config['density'][cwp]
            assert type(cwp_val) == int, f"density.{cwp} must be an integer: {cwp_val}"
    for cwp in ['minFraction', 'binSize', 'bw']:
        if cwp in config['density']:
            cwp_val = config['density'][cwp]
            assert type(cwp_val) in [float, int], f"density.{cwp} must be a float: {cwp_val}"
            config['density'][cwp] = float(cwp_val)
    return config

################################################################################
def validate_grouping(config):
    if 'grouping_steps' in config:
        cwp = config['grouping_steps']
        assert type(cwp) == int, f"grouping_steps must be an integer: {cwp}"
    return config


################################################################################
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = {k: v for k, v in config.items() if v != None}
    for param_group in ['centwave', 'obiwarp', 'density', 'grouping_steps']:
        vf = {'centwave': validate_centwave,
            'obiwarp': validate_obiwarp,
            'density': validate_density,
            'grouping_steps': validate_grouping}.get(param_group, lambda x : x)
        config = vf(config)
    return config

################################################################################
def create_output_directory(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

################################################################################
def read_samples(samples_file):
    if samples_file.endswith('.tsv'):
        sep = '\t'
        tabular = True
    elif samples_file.endswith('.csv'):
        sep = ','
        tabular = True
    elif samples_file.endswith('.xlsx'):
        sep = None
        tabular = False
    else:
        raise ValueError(f"Samples file must be either a .tsv, .csv, or .xlsx file: {samples_file}")
    df_samples = pd.read_csv(samples_file, sep=sep) if tabular else pd.read_excel(samples_file)

    # Check the number of columns
    num_columns = df_samples.shape[1]
    if num_columns not in [2, 3]:
        raise ValueError(f"Samples file must have either 2 or 3 columns, found {num_columns} columns")
    
    if num_columns == 3:
        expected_headers = ['Sample Name', 'Sample Group', 'File']
    else:
        expected_headers = ['Sample Name', 'File']

    # Check for headers or assign them if missing
    if list(df_samples.columns) != expected_headers:
        if list(df_samples.columns) == list(range(num_columns)):
            # If there are no headers, assign the expected headers
            df_samples.columns = expected_headers
        else:
            # If headers are present but incorrect, raise an error
            raise ValueError(f"Column headers must be {expected_headers}, found {list(df_samples.columns)}")
    
    return df_samples

################################################################################

################################################################################
def run_xcms(params, samples, out_file):
    # Extract configuration values
    in_files = ro.StrVector(samples['File'].values)
    sample_names = ro.StrVector(samples['Sample Name'].values)
    sample_groups = ro.StrVector(samples['Sample Group'].values) if 'Sample Group' in samples.columns else np.repeat(0, len(sample_names))

    # Create data frame for sample data
    pheno_data = ro.DataFrame({'sample_name': sample_names, 'sample_group': sample_groups})

    BiocParallel.register(BiocParallel.SerialParam())

    raw_data = msnbase.readMSData(files=in_files, pdata=methods.new('AnnotatedDataFrame', pheno_data), mode='onDisk')
    cwp = xcms.CentWaveParam(
        ppm = params.get('centwave', {}).get('ppm', 30.0),
        mzdiff = params.get('centwave', {}).get('mzdiff', 0.01),
        integrate = params.get('centwave', {}).get('integrate', 1),
        fitgauss = params.get('centwave', {}).get('fitgauss', False),
        noise = params.get('centwave', {}).get('noise', 0.0),
        peakwidth = ro.FloatVector(params.get('centwave', {}).get('peakwidth', [10.0, 60.0])),
        prefilter = ro.IntVector(params.get('centwave', {}).get('prefilter', [3.0, 500.0])),
        snthresh = params.get('centwave', {}).get('snthresh', 10.0),
        mzCenterFun = params.get('centwave', {}).get('mzCenterFun', 'wMean')
    )

    xdata = xcms.findChromPeaks(raw_data, param=cwp)

    owp = xcms.ObiwarpParam(
        factorGap = params.get('obiwarp', {}).get('factorGap', 1.0),
        binSize = params.get('obiwarp', {}).get('binSize', 0.5),
        factorDiag = params.get('obiwarp', {}).get('factorDiag', 2.0),
        distFun = params.get('obiwarp', {}).get('distFun', 'cor_opt'),
        response = params.get('obiwarp', {}).get('response', 1.0),
        localAlignment = params.get('obiwarp', {}).get('localAlignment', False),
        initPenalty = params.get('obiwarp', {}).get('initPenalty', 0.0)
    )
    pheno_data = xdata.slots['phenoData']
    pheno_data_df = pheno_data.slots['data']
    sample_groups = pheno_data_df['sample_group']
    sample_groups_r = ro.StrVector(sample_groups)
    pdp = xcms.PeakDensityParam(
        sampleGroups=sample_groups_r,
        minSamples = params.get('density', {}).get('minSamples', 1),
        minFraction = params.get('density', {}).get('minFraction', 0.25),
        binSize = params.get('density', {}).get('binSize', 0.025),
        bw = params.get('density', {}).get('bw', 3.0),
        maxFeatures = params.get('density', {}).get('maxFeatures', 200)
    )

    grouping_steps = params.get('grouping_steps', 3)
    for gs in range(grouping_steps):
        xdata = xcms.adjustRtime(xdata, param=owp)
        xdata = xcms.groupChromPeaks(xdata, param=pdp)
    xdata = xcms.fillChromPeaks(xdata)

    fd = xcms.featureDefinitions(xdata)
    fv = xcms.featureValues(xdata, value='into')

    pheno_data = biobase.phenoData(xdata)
    cp = xcms.chromPeaks(xdata)
    sample_names = pheno_data.slots['data']['sample_name']

    list_data = fd.do_slot('listData')
    df_fd = pd.DataFrame(dict(zip(list_data.names, map(list,list(list_data)))))
    df_fd['peakidx'] = df_fd['peakidx'].apply(lambda x : np.array(x).astype(int))
    df_fd.insert(0, "feature_id", fd.slots['rownames'])
    peakidxs = df_fd['peakidx'].values
    df_pd = pheno_data.slots['data']
    df_pd.index = df_pd.index.astype(int)
    df_fv = pd.DataFrame(fv, columns = df_pd.sample_name)
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
    ro.r['saveRDS'](xdata, file=out_tsv.replace('.tsv', '_XCMSSET.rds'))


################################################################################
def main():
    args = parse_arguments()
    load_R_libraries()
    params = read_config(args.parameters)
    samples = read_samples(args.samples)
    create_output_directory(out_dir = Path(args.output).parent)
    # run_xcms(samples, params, args.output)

if __name__ == "__main__":
    main()
