import pandas as pd
import argparse
import logging
import traceback
from pathlib import Path
import yaml
import numpy as np
import ipdb
from .utils import create_output_directory, read_samples

################################################################################
def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Run XCMS on input data")
    parser.add_argument('--parameters', '-p', required=True, help='YAML configuration file with parameters')
    parser.add_argument('--samples', '-s', required=True, help='tsv/csv/xlsx file containing input sample information')
    parser.add_argument('--out_file', '-o', required=True, help='Path to output tsv or csv file')
    parser.add_argument('--no_logfile', '-n', action='store_true', help='Disable saving log to a file')
    parser.add_argument('--debug_files', '-d', action='store_true', help='Save output .rds and chromPeak files for debugging')
    parser.add_argument('--ipdb_debug', action='store_true', help='Enable debugging with ipdb')
    return parser

################################################################################
def parse_arguments(parser):
    args = parser.parse_args()
    if not (Path(args.out_file).suffix == '.tsv' or Path(args.out_file) == '.csv'):
        raise ValueError(f"Output file must be a .tsv or .csv file: {args.out_file}")
    return args

################################################################################
def load_R_libraries():
    # Loading R libraries globally from within function in order to perform argparse validation before loading R
    global ro, pandas2ri, base, utils, msnbase, xcms, dplyr, stringr, BiocParallel, stats, methods, biobase
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
def run_xcms(args):
    in_files = ro.StrVector(args.in_files)
    sample_names = ro.StrVector(args.sample_names)
    sample_groups = ro.StrVector(args.sample_groups)

    # Create data frame for sample data
    pheno_data = ro.DataFrame({'sample_name': sample_names, 'sample_group': sample_groups})

    BiocParallel.register(BiocParallel.SerialParam())

    raw_data = msnbase.readMSData(files=in_files, pdata=methods.new('AnnotatedDataFrame', pheno_data), mode='onDisk')
    cwp = xcms.CentWaveParam(
        ppm = args.params.get('centwave', {}).get('ppm', 30.0),
        mzdiff = args.params.get('centwave', {}).get('mzdiff', 0.01),
        integrate = args.params.get('centwave', {}).get('integrate', 1),
        fitgauss = args.params.get('centwave', {}).get('fitgauss', False),
        noise = args.params.get('centwave', {}).get('noise', 0.0),
        peakwidth = ro.FloatVector(args.params.get('centwave', {}).get('peakwidth', [10.0, 60.0])),
        prefilter = ro.IntVector(args.params.get('centwave', {}).get('prefilter', [3.0, 500.0])),
        snthresh = args.params.get('centwave', {}).get('snthresh', 10.0),
        mzCenterFun = args.params.get('centwave', {}).get('mzCenterFun', 'wMean')
    )

    xdata = xcms.findChromPeaks(raw_data, param=cwp)

    owp = xcms.ObiwarpParam(
        factorGap = args.params.get('obiwarp', {}).get('factorGap', 1.0),
        binSize = args.params.get('obiwarp', {}).get('binSize', 0.5),
        factorDiag = args.params.get('obiwarp', {}).get('factorDiag', 2.0),
        distFun = args.params.get('obiwarp', {}).get('distFun', 'cor_opt'),
        response = args.params.get('obiwarp', {}).get('response', 1.0),
        localAlignment = args.params.get('obiwarp', {}).get('localAlignment', False),
        initPenalty = args.params.get('obiwarp', {}).get('initPenalty', 0.0)
    )
    pheno_data = xdata.slots['phenoData']
    pheno_data_df = pheno_data.slots['data']
    sample_groups = pheno_data_df['sample_group']
    sample_groups_r = ro.StrVector(sample_groups)
    pdp = xcms.PeakDensityParam(
        sampleGroups=sample_groups_r,
        minSamples = args.params.get('density', {}).get('minSamples', 1),
        minFraction = args.params.get('density', {}).get('minFraction', 0.25),
        binSize = args.params.get('density', {}).get('binSize', 0.025),
        bw = args.params.get('density', {}).get('bw', 3.0),
        maxFeatures = args.params.get('density', {}).get('maxFeatures', 200)
    )

    grouping_steps = args.params.get('grouping_steps', 3)
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
        # Remember that here we changed the reported mz_ and rt_ to be the mean across a sample instead of individually reporting each intra-sample chromatographic peak
        df_cp_x_summary = df_cp_x.groupby('sample_name').mean()[['mz', 'rt']]
        row = {f"rt_{k}" : v for k,v in df_cp_x_summary['rt'].to_dict().items()}
        row.update({f"mz_{k}" : v for k,v in df_cp_x_summary['mz'].to_dict().items()})
        peak_description_rows.append(row)
    df_peak_description = pd.DataFrame(peak_description_rows)
    # for sn in df_pd['sample_name']:
    #     if sn not in df_peak_description.columns:
    #         df_peak_description[sn] = None
    def sort_fun(x):
        sort_val = list(df_pd['sample_name'].values).index(x.replace('rt_', '').replace('mz_', ''))
        if 'mz_' in x:
            sort_val += len(df_pd) + 1
        return sort_val
    df_peak_description = df_peak_description[sorted(df_peak_description.columns, key = lambda x : sort_fun(x))]
    df_merged = pd.concat([df_fd, df_fv, df_peak_description], axis = 1)
    df_merged.insert(0, "name", df_merged.apply(lambda row: f"M{round(row['mzmed'])}T{round(row['rtmed'])}", axis=1))
    name_counts = dict(zip(*np.unique(df_merged['name'], return_counts=True)))
    suf_counts = {k : 1 for k,v in name_counts.items() if v > 1}
    non_redun_names = []
    for i_name, name in enumerate(df_merged['name']):
        if name in suf_counts:
            non_redun_names.append(f"{name}_{suf_counts[name]}")
            suf_counts[name] += 1
        else:
            non_redun_names.append(name)
    df_merged['name'] = non_redun_names

    if Path(args.out_file).suffix == '.tsv':
        out_sep = '\t'
    elif Path(args.out_file).suffix == '.csv':
        out_sep = ','
    else:
        raise ValueError(f"Output file must be either a .tsv or .csv file: {args.out_file}")
    df_merged.to_csv(args.out_file, sep=out_sep, index=False)
    if args.debug_files:
        df_cp.to_csv(Path(args.out_file).with_suffix('.CPDATA.tsv'), sep='\t', index=False)
        ro.r['saveRDS'](xdata, file=str(Path(args.out_file).with_suffix('.XCMSSET.rds')))


################################################################################
def main(args):
    try:
        load_R_libraries()
        args.params = read_config(args.parameters)
        df_samples, in_files, sample_names, sample_groups, _ = read_samples(args.samples)
        args.in_files = in_files
        args.sample_names = sample_names
        args.sample_groups = sample_groups
        args.out_dir = Path(args.out_file).parent
        create_output_directory(out_dir = args.out_dir)
        args.log_filename = args.out_dir / f"{Path(args.out_file).stem}.log"
        logging_handlers = [logging.StreamHandler()]
        if not args.no_logfile:
            logging_handlers.append(logging.FileHandler(args.log_filename))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=logging_handlers)

        # Log the parameters and samples
        logging.info(f"Args: {args}")    
        run_xcms(args)
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        traceback.print_exc()  # This will print the traceback to the console
        if args.ipdb_debug:
            ipdb.post_mortem()

if __name__ == "__main__":
    parser = get_parser()
    args = parse_arguments(parser)
    main(args)
