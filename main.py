from models.facebook_bart_large_mnli import run_test
from processes.preprocess import preprocess, get_cs_tickets_df
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('-rp', '--run-preprocess', action='store_true', help='Runs preprocessor and save report')
args = parser.parse_args()


def main():
    if args.run_preprocess:
        preprocess()

    df = get_cs_tickets_df()
    run_test(df)

if __name__ == "__main__":
    main()
