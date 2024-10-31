import argparse
import logging
import traceback
from pathlib import Path
import yaml
from .utils import create_output_directory
from . import feature_finding, clustering, stats_tests

################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Runs the entire metabozen pipeline using a provided yaml configuration file.")
    parser.add_argument('parameters', help='YAML configuration file with parameters')
    parser.add_argument('--no_logfile', '-n', action='store_true', help='Disable saving log.')
    args = parser.parse_args()
    return args

################################################################################
def validate_params(config):
    assert 'samples' in config, "`samples` (input samples files) must be provided in the configuration file"
    assert Path(config['samples']).exists(), f"Samples file does not exist: {config['samples']}"
    assert 'out_dir' in config, "`out_dir` (output directory) must be provided in the configuration file"
    assert 'run_name' in config, "`run_name` must be provided in the configuration file"
    for step in ['feature_finding', 'clustering']:
        assert step in config, f"`{step}` must be provided in the configuration file"
        assert 'parameters' in config[step], f"`parameters` must be provided for `{step}` in the configuration file"
        assert Path(config[step]['parameters']).exists(), f"Parameters file for `{step}` does not exist: {config[step]['parameters']}"
    if 'stats_tests' in config:
        assert 'parameters' in config['stats_tests'], "To run the optional `stats_tests` module, parameters` must be provided for `stats_tests` in the configuration file"
        assert Path(config['stats_tests']['parameters']).exists(), "Parameters file for `stats_tests` does not exist"
        if 'plot_fomat' not in config['stats_tests']:
            config['stats_tests']['plot_format'] = 'png'
    return config

################################################################################
def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = {k: v for k, v in config.items() if v != None}
    config = validate_params(config)
    return config

################################################################################
def make_feature_finding_args(args):
    out_args = argparse.Namespace()
    out_args.parameters = args.config['feature_finding']['parameters']
    out_args.samples = args.config['samples']
    out_args.out_file = Path(args.config['out_dir']) / args.config['run_name'] / args.config['feature_finding']['out_file']
    for opt_param in ['debug_files']:
        if opt_param in args.config['feature_finding']:
            setattr(out_args, opt_param, args.config['feature_finding'][opt_param])
        else:
            parser = feature_finding.get_parser()
            setattr(out_args, opt_param, parser.get_default(opt_param))
    if 'no_logfile' in args:
        if args.no_logfile:
            parser = feature_finding.get_parser()
            setattr(out_args, 'no_logfile', parser.get_default('no_logfile'))
        else:
            out_args.no_logfile = True
    return out_args

################################################################################
def make_clustering_args(args):
    out_args = argparse.Namespace()
    out_args.parameters = args.config['clustering']['parameters']
    out_args.samples = args.config['samples']
    out_args.out_file = Path(args.config['out_dir']) / args.config['run_name'] / args.config['clustering']['out_file']
    out_args.xcms_in_file = Path(args.config['out_dir']) / args.config['run_name'] / args.config['feature_finding']['out_file']
    for opt_param in ['debug_files']:
        if opt_param in args.config['clustering']:
            setattr(out_args, opt_param, args.config['clustering'][opt_param])    
        else:
            parser = clustering.get_parser()
            setattr(out_args, opt_param, parser.get_default(opt_param))
    if 'no_logfile' in args:
        if args.no_logfile:
            parser = clustering.get_parser()
            setattr(out_args, 'no_logfile', parser.get_default('no_logfile'))
        else:
            out_args.no_logfile = True
    return out_args

################################################################################
def make_stats_tests_args(args):
    out_args = argparse.Namespace()
    out_args.parameters = args.config['stats_tests']['parameters']
    out_args.samples = args.config['samples']
    out_args.out_file = Path(args.config['out_dir']) / args.config['run_name'] / args.config['stats_tests']['out_file']
    out_args.clustering_in_file = Path(args.config['out_dir']) / args.config['run_name'] / args.config['clustering']['out_file']
    for opt_param in ['no_plots', 'plot_format']:
        if opt_param in args.config['stats_tests']:
            setattr(out_args, opt_param, args.config['stats_tests'][opt_param])    
        else:
            parser = stats_tests.get_parser()
            setattr(out_args, opt_param, parser.get_default(opt_param))
    if 'no_logfile' in args:
        if args.no_logfile:
            parser = stats_tests.get_parser()
            setattr(out_args, 'no_logfile', parser.get_default('no_logfile'))
        else:
            out_args.no_logfile = True
    return out_args

################################################################################
def main():
    try:
        args = parse_arguments()
        args.config = read_config(args.parameters)
        create_output_directory(Path(args.config['out_dir']) / args.config['run_name'])
        args.log_filename = Path(args.config['out_dir']) / f"{args.config['run_name']}.log"
        logging_handlers = [logging.StreamHandler()]
        logging_handlers.append(logging.FileHandler(args.log_filename))
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=logging_handlers)
        # Log the parameters and samples
        logging.info(f"Args: {args}")
        feature_finding_args = make_feature_finding_args(args)
        logging.info(f"Feature Finding Args: {feature_finding_args}")
        clustering_args = make_clustering_args(args)
        logging.info(f"Clustering Args: {clustering_args}")
        if 'stats_tests' in args.config:
            stats_tests_args = make_stats_tests_args(args)
            logging.info(f"Stats Tests Args: {stats_tests_args}")
        logging.info("Starting feature finding")
        feature_finding.main(feature_finding_args)
        logging.info("Starting clustering")
        clustering.main(clustering_args)
        if 'stats_tests' in args.config:
            logging.info("Starting stats tests")
            stats_tests.main(stats_tests_args)
        logging.info("Pipeline complete")
    
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        traceback.print_exc()  # This will print the traceback to the console

if __name__ == "__main__":
    main()
