import argparse
import yaml
import logging
import traceback
import scipy.stats
from tqdm.auto import tqdm
import ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import read_samples, create_output_directory

################################################################################
def get_parser():
    parser = argparse.ArgumentParser(description="Clustering script for XCMS output data")
    parser.add_argument('--clustering_in_file', '-i', required=True, help='tsv/csv file produced from XCMS feature finding run.')
    parser.add_argument('--parameters', '-p', required=True, help='YAML configuration file with parameters')
    parser.add_argument('--samples', '-s', required=True, help='tsv/csv/xlsx file containing input sample information')
    parser.add_argument('--out_file', '-o', required=True, help='Path to output tsv or csv file. Tsv is recommended. Parent directory will be created if nonexistent.')
    parser.add_argument('--no_logfile', '-n', action='store_true', help='Disable saving log to a file')
    parser.add_argument('--no_plots', '-np', action='store_true', help='Disable saving plots')
    parser.add_argument('--plot_format', default="png", help="Format for saving plots (png, svg, etc.)", choices=['png', 'svg', 'jpg', 'jpeg'])
    parser.add_argument('--ipdb_debug', action='store_true', help='Enable debugging with ipdb')
    return parser
################################################################################
def parse_arguments(parser):
    args = parser.parse_args()
    if not (Path(args.out_file).suffix == '.tsv' or Path(args.out_file).suffix == '.csv'):
        raise ValueError(f"Output file must be a .tsv or .csv file: {args.out_file}")
    if not (Path(args.clustering_in_file).suffix == '.tsv' or Path(args.clustering_in_file).suffix == '.csv'):
        raise ValueError(f"Clustering input file must be a .tsv or .csv file: {args.clustering_in_file}")
    return args

################################################################################
def validate_mann_whitney_u(config):
    muc = config['mann_whitney_u']
    assert 'group_1' in muc, "Missing 'group_1' in Mann-Whitney U test parameters"
    assert 'group_2' in muc, "Missing 'group_2' in Mann-Whitney U test parameters"
    if 'kwargs' in muc:
        muck = muc['kwargs']
        if 'paired' in muck:
            assert isinstance(muck['paired'], bool), f"Paired parameter must be a boolean: {muck['paired']}"
        else:
            muc['kwargs']['paired'] = False
        if 'qvalues' in muck:
            assert isinstance(muck['qvalues'], bool), f"qvalues parameter must be a boolean: {muck['qvalues']}"
            load_R_libraries()
        else:
            muc['kwargs']['qvalues'] = True
    config['mann_whitney_u'] = muc
    return config

################################################################################
def validate_params(config):
    stats_tests =  {'mann_whitney_u' : validate_mann_whitney_u}
    test_names = list(stats_tests.keys())
    for tn in config:
        if tn not in test_names:
            raise ValueError(f"Invalid test {tn} in parameters file. Valid test names are: {test_names}")
        config = stats_tests[tn](config)
    return config

################################################################################
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = {k: v for k, v in config.items() if v != None}
    config = validate_params(config)
    return config

################################################################################
def load_R_libraries():
    global ro, qvalue
    import rpy2.robjects as ro
    import rpy2.robjects.packages as packages
    rbase = packages.importr('base')
    qvalue = packages.importr('qvalue')
    return rbase, qvalue

################################################################################
def read_clustering_file(args):
    if Path(args.clustering_in_file).suffix == '.tsv':
        in_sep = '\t'
    elif Path(args.clustering_in_file).suffix == '.csv':
        in_sep = ','
    else:
        raise ValueError(f"Clustering input file must be either a .tsv or .csv file: {args.clustering_in_file}")
    df_clusters = pd.read_csv(args.clustering_in_file, sep='\t')
    return df_clusters

################################################################################
def make_plot(vals, title, fname_out):
        fig, ax = plt.subplots(figsize=(9, 6))
        plt.hist(vals, color='lightsteelblue', edgecolor='black', bins=20)
        plt.title(title)
        if Path(fname_out) in ['.png', '.jpg', '.jpeg']:
            plt.savefig(fname_out, dpi=400)
        else:
            plt.savefig(fname_out)

################################################################################
def run_mann_whitney_u(args):
    sns_g1 = args.df_samples.loc[lambda x : x['Sample Group'] == args.config["mann_whitney_u"]['group_1']]['Sample Name'].values
    sns_g2 = args.df_samples.loc[lambda x : x['Sample Group'] == args.config["mann_whitney_u"]['group_2']]['Sample Name'].values
    logging.info(args.config["mann_whitney_u"])
    if args.config["mann_whitney_u"]['kwargs']['paired']:
        assert len(sns_g1) == len(sns_g2), f"Paired Mann-Whitney U test requires equal number of samples in both groups: {len(sns_g1)} vs {len(sns_g2)}"
    df_out = args.df_clusters.copy()
    stats = ['not calc' for i in range(len(df_out))]
    pvals = ['not calc' for i in range(len(df_out))]
    for i in tqdm(range(len(df_out))):
        if (df_out.iloc[i][sns_g1].isnull().all() & df_out.iloc[i][sns_g2].isnull().all()): 
            stat, pval = scipy.stats.mannwhitneyu([1e-11, 0, 0], [1e-11, 0], alternative = 'two-sided')
        elif not args.config['mann_whitney_u']['kwargs']['paired']:
            stat, pval = scipy.stats.mannwhitneyu(
                df_out.iloc[i][sns_g1].astype(float).fillna(value = 0.0), 
                df_out.iloc[i][sns_g2].astype(float).fillna(value = 0.0), 
                alternative = 'two-sided')
        else:
            stat, pval = scipy.stats.wilcoxon(df_out.iloc[i][sns_g1].astype(float).fillna(value = 0.0),
                                              df_out.iloc[i][sns_g2].astype(float).fillna(value = 0.0),
                                              mode = 'exact')
        stats[i] = stat
        pvals[i] = pval
    out_cols = ['mann_whitney_u_muStats', 'mann_whitney_u_muPvals']
    # Dictionary of {test_name : plot_title}
    # plot_cols = {'mann_whitney_u_muPvals' : "P-values Mann-Whitney all features"}
    df_out['mann_whitney_u_muStats'] = stats
    df_out['mann_whitney_u_muPvals'] = pvals
    if args.config['mann_whitney_u']['kwargs']['qvalues']:
        out_cols.append('mann_whitney_u_muQvals')
        # plot_cols['mann_whitney_u_muQvals'] = "Q-values Mann-Whitney parent features"
        df_parents = df_out[df_out.apply(lambda x: x['name'] == x['parent_peak'], axis=1)]
        try:
            qresult = qvalue.qvalue(p = ro.FloatVector(df_parents['mann_whitney_u_muPvals'].values))
            df_qresults = pd.DataFrame(np.array(qresult.rx('qvalues')[0]), index = df_parents.index)
            df_out['mann_whitney_u_muQvals'] = df_qresults.reindex(df_out.index, fill_value = None).values[:, 0]
        except Exception as e:
            logging.error(f"Error calculating q-values: {e}")
            df_out['mann_whitney_u_muQvals'] = [0.0 for i in range(len(df_out))]
    return pd.DataFrame(df_out[out_cols])

################################################################################
def plot_mann_whitney_u(df_statistics, args):
    plot_vals = [df_statistics['mann_whitney_u_muPvals'], df_statistics.dropna(subset = ['mann_whitney_u_muQvals'])['mann_whitney_u_muPvals'], df_statistics.dropna(subset = ['mann_whitney_u_muQvals'])['mann_whitney_u_muQvals']]
    plot_titles = ['P-values Mann-Whitney all features', 'P-values Mann-Whitney parent features', 'Q-values Mann-Whitney parent features']
    fnames = [args.out_dir / f"{x}.{args.plot_format}" for x in ['mann_whitney_u_muPvals_allFeats', 'mann_whitney_u_muPvals_parentFeats', 'mann_whitney_u_muQvals_parentFeats']]
    for vals, title, fname_out in zip(plot_vals, plot_titles, fnames):
        make_plot(vals = vals, title = title, fname_out = fname_out)

################################################################################
def run_test(test_name, args):
    stats_tests = {'mann_whitney_u' : {'run' : run_mann_whitney_u, 'plot' : plot_mann_whitney_u}}
    if test_name not in stats_tests:
        raise ValueError(f"Invalid test name {test_name}. Valid test names are: {list(stats_tests.keys())}")
    logging.info(f"Running test - {test_name}")
    df_statistics = stats_tests[test_name]['run'](args)
    if not args.no_plots:
        logging.info(f"Plotting test - {test_name}")
        stats_tests[test_name]['plot'](df_statistics, args)
    return df_statistics    

################################################################################
def run_statistics(args):
    data_cols = list(args.sample_names) + [f"rt_{x}" for x in args.sample_names] + [f"mz_{x}" for x in args.sample_names]
    df_data = args.df_clusters[data_cols]
    df_header = args.df_clusters[[x for x in args.df_clusters.columns if x not in data_cols]]
    test_results = []
    for test_name in args.config:
        df_test = run_test(test_name, args)
        test_results.append(df_test)
    df_statistics = pd.concat([df_header] + test_results + [df_data], axis=1)
    return df_statistics    

################################################################################
def main(args):
    try:
        args.config = read_config(args.parameters)
        args.df_clusters = read_clustering_file(args)
        df_samples, _, sample_names, sample_groups, _ = read_samples(args.samples)
        args.df_samples = df_samples
        args.sample_names = sample_names
        args.sample_groups = sample_groups
        args.out_dir = Path(args.out_file).parent
        create_output_directory(args.out_dir)
        args.log_filename = args.out_dir / f"{Path(args.out_file).stem}.log"
        logging_handlers = [logging.StreamHandler()]
        if not args.no_logfile:
            logging_handlers.append(logging.FileHandler(args.log_filename))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=logging_handlers)
        logging.info(f"Args: {args}")
        args.df_statistics = run_statistics(args)
        if Path(args.out_file).suffix == '.tsv':
            out_sep = '\t'
        elif Path(args.out_file).suffix == '.csv':
            out_sep = ','
        args.df_statistics.to_csv(args.out_file, sep=out_sep, index=False)
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        traceback.print_exc()  # This will print the traceback to the console
        if args.ipdb_debug:
            ipdb.post_mortem()

if __name__ == "__main__":
    parser = get_parser()
    args = parse_arguments(parser)
    main(args)