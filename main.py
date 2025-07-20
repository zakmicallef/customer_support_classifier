from processes.preprocess import preprocess
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('-rp', '--run-preprocess', action='store_true', help='Runs preprocessor and save report')
args = parser.parse_args()


def main():
    if args.run_preprocess:
        preprocess()

if __name__ == "__main__":
    main()
