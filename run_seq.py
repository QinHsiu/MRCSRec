import argparse

from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MRCSRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Yelp', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,do_eval=args.do_eval)