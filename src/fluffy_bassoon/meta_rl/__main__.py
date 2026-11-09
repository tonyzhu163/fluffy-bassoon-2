"""Fit a history policy from train and validation trajectory archives."""
import argparse
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s METHOD [training options]")
    parser.add_argument("method", choices=("ad", "dpt", "dit", "ic_iql"),
                        help="use METHOD --help for its training options")
    method = parser.parse_args(argv[:1]).method
    if method in ("ad", "dpt"):
        from .vla_history_train import main as fit
    else:
        from .history_value_train import main as fit
    return fit(argv[1:], objective=method)


if __name__ == "__main__":
    main()
