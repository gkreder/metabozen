from pathlib import Path
import logging
import yaml
import scipy.spatial
import scipy.cluster
from tqdm.auto import tqdm
import pandas as pd
import argparse
import copy
import sklearn.metrics
import numpy as np
import ast
import warnings
from utils import create_output_directory, read_samples

################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Clustering script for XCMS output data")
    parser.add_argument('--parameters', '-p', required=True, help='YAML configuration file with parameters')
    parser.add_argument('--samples', '-s', required=True, help='tsv/csv/xlsx file containing input sample information')
    parser.add_argument('--out_file', '-o', required=True, help='Path to output tsv or csv file. Tsv is recommended. Parent directory will be created if nonexistent.')
    parser.add_argument('--no_logfile', '-n', action='store_true', help='Disable saving log to a file')
    parser.add_argument('--xcms_in_file', '-i', required=True, help='tsv/csv file produced from XCMS feature finding run.')
    parser.add_argument('--debug_files', action='store_true', help='Save intermediate files (including intermediate recursive clustering output) for debugging')
    args = parser.parse_args()
    if not (args.out_file.endswith('.tsv') or args.out_file.endswith('.csv')):
        raise ValueError(f"Output file must be a .tsv or .csv file: {args.out_file}")
    return args

################################################################################
def validate_params(config):
    cp_defaults = {
        "alpha": 0.25,
        "tau": 4.0,
        "frac_peaks": 0.8,
        "r1_1sWindow": 5.0,
        "cluster_outlier_1sWidth": 3.0,
        "rt_iqr_filter": 1.5,
        "parent_mz_check_intensity_frac": 0.6,
        "dropped_clust_RT_width": 2.5,
        "recursive_clustering": True
    }
    for cp in ["alpha", "tau", "frac_peaks", "r1_1sWindow", "cluster_outlier_1sWidth", "rt_iqr_filter", "parent_mz_check_intensity_frac", "dropped_clust_RT_width"]:
        if cp in config:
            cpv = config[cp]
            assert (type(cpv) == float) and (cpv >= 0.0), f"{cp} must be a float >= 0: {cpv}"
        else:
            config[cp] = cp_defaults[cp]
    if "recursive_clustering" in config:
        cp = "recursive_clustering"
        cpv = config[cp]
        assert type(cpv) == bool, f"recursive_clustering must be a boolean: {cpv}"
    else:
        config["recursive_clustering"] = cp_defaults["recursive_clustering"]
    if 'recursive_safety_limit' in config:
        cp = 'recursive_safety_limit'
        cpv = config[cp]
        assert type(cpv) == int and cpv > 0, f"recursive_safety_limit must be an integer > 0: {config['recursive_safety_limit']}"
    else:
        config['recursive_safety_limit'] = 100
    return config

################################################################################
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = {k: v for k, v in config.items() if v != None}
    config = validate_params(config)
    return config

################################################################################
def  read_xcms_input(args):
    df = pd.read_csv(args.xcms_in_file, sep='\t').set_index('name')
    intensity_ranks = df[args.sample_names].mean(axis=1, skipna=True).sort_values(ascending=False).index
    df_sorted_intensities = df[args.sample_names].loc[intensity_ranks]
    rt_cols = [f"rt_{x}" for x in args.sample_names]
    df_sorted_rts = df[rt_cols].loc[intensity_ranks]
    df_sorted_intensities = np.log10(df_sorted_intensities.applymap(merge_func).replace(to_replace=0, value=1.0))
    df_sorted_rts = df_sorted_rts.applymap(merge_func)
    return df, df_sorted_intensities, df_sorted_rts

def merge_func(x):
    try:
        m = np.mean(ast.literal_eval(x))
    except:
        m = x
    return m

################################################################################
def calculate_linkage(args):
    logging.info("Calculating linkage")
    def get_dist_mat(df_sorted_intensities, df_sorted_rts, max_distance):
        if df_sorted_intensities.shape != df_sorted_rts.shape:
            raise ValueError(f"Error in dist_mat - df_sorted_intensities shape {df_sorted_intensities.shape} must be equal to df_sorted_rts shape {df_sorted_rts.shape}")

        num_cols = df_sorted_rts.shape[1]
        df_noNans = (~df_sorted_intensities.isna()).astype(int).dot((~df_sorted_intensities.isna()).astype(int).T)
        df_bothNans = (df_sorted_intensities.isna()).astype(int).dot((df_sorted_intensities.isna()).astype(int).T)
        
        pDists = (1 - df_sorted_intensities.T.corr(method='pearson')).values
        mDists = sklearn.metrics.pairwise.manhattan_distances(df_sorted_intensities.fillna(value=0.0))

        pDists[np.where((df_noNans > 0) & (df_noNans < 3) & ((num_cols - df_noNans) <= df_bothNans) & (mDists < 1e-2))] = 0.0
        pDists[np.where((df_noNans > 0) & (df_noNans < 3) & ((num_cols - df_noNans) <= df_bothNans) & (mDists >= 1e-2))] = 0.1

        pDists[np.where(np.isnan(pDists) & (mDists < 1e-2))] = 0.0
        pDists[np.where(np.isnan(pDists) & (mDists >= 1e-2))] = max_distance / 2.0

        rDists = np.round(np.sqrt(sklearn.metrics.pairwise.nan_euclidean_distances(df_sorted_rts.values, squared=True) / num_cols), 3)
        if args.params["tau"] == 0.0:
            rDists = (np.ones(rDists.shape) - (np.nan_to_num(np.abs(rDists), nan=np.inf) <= 1e-4).astype(int)) * args.params["alpha"]
        else:
            rDists = args.params["alpha"] * (1 - np.exp(-rDists / args.params["tau"]))
        
        allDists = pDists + rDists
        allDists[np.where(df_noNans == 0)] = max_distance
        allDists[np.where((df_noNans > 0) & (df_noNans < 3) & ((num_cols - df_noNans) > df_bothNans))] = max_distance

        return pd.DataFrame(allDists, index=df_sorted_intensities.index, columns=df_sorted_intensities.index)

    dist_mat = get_dist_mat(args.df_sorted_intensities, args.df_sorted_rts, args.max_distance)
    dist_diffs = dist_mat - dist_mat.T
    max_diff = dist_diffs.to_numpy().max()
    if max_diff > 1e-4:
        max_row = dist_diffs.max().sort_values(ascending=False).index[0]
        max_col = dist_diffs[max_row].sort_values(ascending=False).index[0]    
        raise ValueError(f"Distance matrix calculation error: check calculation at {max_row}, {max_col} with difference of {dist_diffs[max_row][max_col]}")
    if np.any(np.diag(dist_mat)):
        raise ValueError("Distance matrix diagonal contains non-zero values")

    dist_mat_1D = scipy.spatial.distance.squareform(dist_mat, checks=False)
    return scipy.cluster.hierarchy.linkage(dist_mat_1D, method='average')

################################################################################
def find_clusters(args):
    logging.info("Finding clusters")
    def rt_distance(rts1, rts2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            r = np.nan_to_num(np.nanmedian(np.abs(rts1 - rts2)), nan=args.params["rt_1sWindow"] + 1.0)
        return r

    def set_rt(leaf, df_sorted_rts):
        rt = df_sorted_rts.iloc[leaf.id]
        leaf.rt = rt

    def get_cuts(tree, rt_1sWindow, frac_peaks):
        leaf_list = sorted(tree.pre_order(lambda leaf: (leaf.id, leaf.rt)), key=lambda tup: tup[0])
        current_rt = leaf_list[0][1]
        good_peaks = np.sum([rt_distance(current_rt, x[1]) <= rt_1sWindow for x in leaf_list]) / len(leaf_list)
        if good_peaks >= frac_peaks:
            parent_id = leaf_list[0][0]
            cluster_ids = [x[0] for x in leaf_list]
            return [(parent_id, cluster_ids)]
        else:
            return get_cuts(tree.get_right(), rt_1sWindow, frac_peaks) + get_cuts(tree.get_left(), rt_1sWindow, frac_peaks)

    tree = scipy.cluster.hierarchy.to_tree(args.linkage)
    tree.pre_order(lambda leaf: set_rt(leaf, args.df_sorted_rts))

    cut_ids = get_cuts(tree, rt_1sWindow=args.params["rt_1sWindow"], frac_peaks=args.params["frac_peaks"])
    clusters = [(args.df_sorted_intensities.index[p_id], p_id, clust_ids, [args.df_sorted_intensities.index[i] for i in clust_ids]) for p_id, clust_ids in cut_ids]
    return pd.DataFrame(clusters, columns=['peak_name', 'sorted_index', 'cluster_subtree_ids', 'cluster_subtree_names'])


################################################################################
def merge_clusters(args):
    logging.info("Merging clusters")
    clusters = []
    for i_p, peak_name in enumerate(tqdm(args.df_cuts['peak_name'])):
        p_clust = args.df_cuts['cluster_subtree_names'].values[i_p]
        clust_size = len(p_clust)
        clust_intensities, clust_rts, clust_mzs = [], [], []
        for p in p_clust:
            clust_mzs.append(args.df.loc[p]['mzmed'])
            clust_intensities.append(args.df_sorted_intensities.loc[p].to_dict())
            clust_rts.append(args.df.loc[p]['rtmed'])
        
        clusters.append((peak_name, p_clust, clust_intensities, clust_mzs, clust_rts, clust_size))

    return pd.DataFrame(clusters, columns=['parent_peak', 'cluster', 'clust_intensities', 'clust_mzs', 'clust_rts', 'clust_size'])


################################################################################
def finalize_clusters(args):
    logging.info("Finalizing clusters")
    clusters = []
    for i_p, parent_peak in enumerate(tqdm(args.df_merged['parent_peak'])):
        clust = args.df_merged['cluster'].values[i_p]
        clust_rts = args.df_merged['clust_rts'].values[i_p]

        parent_index = args.df_merged['cluster'].values[i_p].index(parent_peak)
        parent_rt = clust_rts[parent_index]
        lb = parent_rt - args.params["cluster_outlier_1sWidth"]
        ub = parent_rt + args.params["cluster_outlier_1sWidth"]

        keep_indices = [(r >= lb and r <= ub) for r in clust_rts]

        dropped = [(i, c) for i, c in enumerate(clust) if not keep_indices[i]]

        clust = [c for i, c in enumerate(clust) if keep_indices[i]]
        clust_intensities = [d for i, d in enumerate(args.df_merged['clust_intensities'].values[i_p]) if keep_indices[i]]
        clust_mzs = [d for i, d in enumerate(args.df_merged['clust_mzs'].values[i_p]) if keep_indices[i]]
        clust_rts = [d for i, d in enumerate(clust_rts) if keep_indices[i]]
        clust_size = len(clust)

        clust_avg_intensities = [np.mean(args.df[args.sample_names].fillna(value=0.0).loc[x]) for x in clust]
        parent_index = np.argmax(clust_avg_intensities)
        parent_peak = clust[parent_index]
        parent_mz = clust_mzs[parent_index]

        restart = True
        while restart:
            restart = False
            for i_cm, cm in enumerate(clust_mzs):
                if clust_mzs[i_cm] > parent_mz and clust_avg_intensities[i_cm] >= clust_avg_intensities[parent_index] * args.params["parent_mz_check_intensity_frac"]:
                    parent_index = i_cm
                    parent_peak = clust[i_cm]
                    parent_mz = clust_mzs[i_cm]
                    restart = True
                    break

        clusters.append((parent_peak, clust, clust_intensities, clust_mzs, clust_rts, clust_size))

        for i_dpn, dpn in dropped:
            clusters.append((dpn, [dpn], [args.df_merged['clust_intensities'].values[i_p][i_dpn]],
                             [args.df_merged['clust_mzs'].values[i_p][i_dpn]], [args.df_merged['clust_rts'].values[i_p][i_dpn]], len([dpn])))

    return pd.DataFrame(clusters, columns=['parent_peak', 'cluster', 'clust_intensities', 'clust_mzs', 'clust_rts', 'clust_size'])


################################################################################
def recursive_clustering(args):
    logging.info("Starting recursive clustering")
    args_rec = copy.deepcopy(args)
    args_rec.df_final_clusters = args.df_final_clusters.set_index('parent_peak', drop=False)
    args_rec.parent_peaks = args.df_final_clusters['parent_peak'].values
    num_clusters = len(args_rec.df_final_clusters)
    recursive_round_num = 1
    while recursive_round_num <= args.params["recursive_safety_limit"]:
        logging.info(f"Recursive clustering round {recursive_round_num} - {num_clusters} clusters")
        args_rec.df_sorted_intensities = pd.DataFrame(args_rec.df_sorted_intensities[args_rec.df_sorted_intensities.index.isin(args_rec.parent_peaks)])
        args_rec.df_sorted_rts = pd.DataFrame(args_rec.df_sorted_rts[args_rec.df_sorted_rts.index.isin(args_rec.parent_peaks)])
        args_rec.linkage = calculate_linkage(args_rec)
        args_rec.df_cuts = find_clusters(args_rec)
        args_rec.df_merged = merge_clusters(args_rec)
        df_new_clusters = finalize_clusters(args_rec)
        df_clusts_to_merge = df_new_clusters.loc[lambda x : x['clust_size'] >= 1]
        for i_p, out_parent in enumerate(df_clusts_to_merge['parent_peak']):
            merge_parents = df_clusts_to_merge['cluster'].values[i_p]
            for parent_pn in merge_parents:
                if parent_pn == out_parent:
                    continue
                c = args_rec.df_final_clusters.loc[parent_pn]
                for i_pn, pn in enumerate(c['cluster']):
                    args_rec.df_final_clusters.loc[out_parent]['cluster'].append(pn)
                    args_rec.df_final_clusters.loc[out_parent]['clust_intensities'].append(c['clust_intensities'][i_pn])
                    args_rec.df_final_clusters.loc[out_parent]['clust_mzs'].append(c['clust_mzs'][i_pn])
                    args_rec.df_final_clusters.loc[out_parent]['clust_rts'].append(c['clust_rts'][i_pn])
                    args_rec.df_final_clusters.loc[out_parent, 'clust_size'] = len(args_rec.df_final_clusters.loc[out_parent]['cluster'])
                args_rec.df_final_clusters = args_rec.df_final_clusters.drop(parent_pn)
        if len(args_rec.df_final_clusters) == num_clusters:
            break
        if args.debug_files:
            args_rec.df_final_clusters.to_csv(args.intermediate_files_dir / f"recursive_round_{recursive_round_num}.tsv", sep='\t', index = False)
        num_clusters = len(args_rec.df_final_clusters)
        recursive_round_num += 1
        args_rec.parent_peaks = args_rec.df_final_clusters['parent_peak'].values
    
    logging.info(f"Recursive clustering finished after {recursive_round_num} rounds")
    return args_rec.df_final_clusters
################################################################################
def join_clustered_output(args):
    logging.info("Joining clustered output")
    df_clusters = args.df_final_clusters.copy()
    df_clusters.insert(0, 'cluster_id', [i for i, _ in enumerate(df_clusters['cluster'])])
    s = df_clusters.apply(lambda x: pd.Series(x['cluster']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'name'
    df_joined = df_clusters.join(s).drop(labels=['clust_intensities', 'clust_rts', 'clust_mzs'], axis=1).set_index('name')
    df_joined = df_joined.merge(args.df, how='outer', left_index=True, right_index=True)
    return df_joined

################################################################################
def normalize_intensities(args):
    logging.info("Normalizing intensities")
    normalization_dict = {x[0] : float(x[1]) for x in zip(args.sample_names, args.normalization)}
    df_normalized = args.df_joined.copy()
    for x in args.sample_names:
        df_normalized[x] = args.df_joined[x] * normalization_dict[x]
    return df_normalized

################################################################################
def prettify_output(args):
    logging.info("Prettifying output")
    df_out = args.df_normalized.copy()
    u_cols = [x for x in args.df_normalized.columns if 'Unnamed:' in x]
    if len(u_cols) > 0:
        df_out = df_out.drop(labels=u_cols, axis=1)
    clust_size_index = list(df_out.columns).index('clust_size') + 1
    df_out.insert(clust_size_index, 'rt_mean', args.df_sorted_rts.mean(axis=1).loc[df_out.index])
    df_out.insert(clust_size_index, 'intensity_mean', df_out[args.sample_names].fillna(0.0).mean(axis=1))
    insert_index = list(df_out.columns).index('peakidx') + 1
    for sg in set(args.sample_groups):
        sg_samples = [x for i_x, x in enumerate(args.sample_names) if args.sample_groups[i_x] == sg]
        df_out.insert(insert_index, f'{sg}_mean', df_out[sg_samples].fillna(0.0).mean(axis=1))
    rt_cols = ['rt_' + x for x in args.sample_names]
    mz_cols = ['mz_' + x for x in args.sample_names]
    df_out_temp = pd.DataFrame(df_out[[x for x in df_out.columns if x not in (args.sample_names + rt_cols)]])
    df_out = pd.concat([df_out_temp, pd.DataFrame(df_out[list(args.sample_names) + rt_cols])], axis=1, sort=False)
    df_out = pd.concat([df_out[[x for x in df_out.columns if x not in mz_cols]], df_out[mz_cols]], axis=1)    
    return df_out

################################################################################
def main():
    args = parse_arguments()
    args.params = read_config(args.parameters)
    df_samples, in_files, sample_names, sample_groups, normalization = read_samples(args.samples)
    args.in_files = in_files
    args.sample_names = sample_names
    args.sample_groups = sample_groups
    args.normalization = normalization
    args.out_dir = Path(args.out_file).parent
    create_output_directory(out_dir = args.out_dir)
    args.log_filename = args.out_dir / f"{Path(args.out_file).stem}.log"
    logging_handlers = [logging.StreamHandler()]
    if not args.no_logfile:
        logging_handlers.append(logging.FileHandler(args.log_filename))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=logging_handlers)
    logging.info(f"Args: {args}")
    if args.debug_files:
        args.intermediate_files_dir = args.out_dir / f"{Path(args.out_file).stem}_clustering_files"
        args.intermediate_files_dir.mkdir(exist_ok=True)
    df, df_sorted_intensities, df_sorted_rts =  read_xcms_input(args)
    args.df = df
    args.df_sorted_intensities = df_sorted_intensities
    args.df_sorted_rts = df_sorted_rts
    args.max_distance = 2.0 + args.params["alpha"]
    args.linkage = calculate_linkage(args)
    args.df_cuts = find_clusters(args)
    args.df_merged = merge_clusters(args)
    args.df_final_clusters = finalize_clusters(args)
    if args.params["recursive_clustering"]:
        args.df_final_clusters = recursive_clustering(args)
    else:
        logging.warning("Recursive clustering disabled, skipping")
    args.df_joined = join_clustered_output(args)
    if type(args.normalization) != type(None):
        args.df_normalized = normalize_intensities(args)
    else:
        logging.warning("No normalization values provided, skipping normalization")
        args.df_normalized = args.df_joined.copy()
    args.df_pretty = prettify_output(args)
    if args.out_file.endswith('.tsv'):
        out_sep = '\t'
    elif args.out_file.endswith('.csv'):
        out_sep = ','
    else:
        raise ValueError(f"Output file must be either a .tsv or .csv file: {args.out_file}")
    args.df_pretty.to_csv(args.out_file, sep=out_sep)

if __name__ == "__main__":
    main()
