import sys
import argparse
from . import feature_finding, clustering, stats_tests, main

def main():
    parser = argparse.ArgumentParser(description="MetaboZen: Untargeted LC-MS metabolomics data analysis")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Feature Finding
    ff_parser = subparsers.add_parser('feature-finding', help='Run XCMS feature finding')
    ff_parser = feature_finding.get_parser(ff_parser)

    # Clustering
    clust_parser = subparsers.add_parser('clustering', help='Run metabolite clustering')
    clust_parser = clustering.get_parser(clust_parser)

    # Stats Tests
    stats_parser = subparsers.add_parser('stats-tests', help='Run statistical tests')
    stats_parser = stats_tests.get_parser(stats_parser)

    # Full Pipeline
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete analysis pipeline')
    pipeline_parser.add_argument('parameters', help='YAML configuration file with parameters')
    pipeline_parser.add_argument('--no_logfile', '-n', action='store_true', help='Disable saving log.')

    args = parser.parse_args()

    if args.command == 'feature-finding':
        feature_finding.main(args)
    elif args.command == 'clustering':
        clustering.main(args)
    elif args.command == 'stats-tests':
        stats_tests.main(args)
    elif args.command == 'pipeline':
        main.main()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 