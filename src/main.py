# from api import run_fast
from processes.test import run_test
from models.facebook_bart_large_mnli import MnliModel
from processes.ingest import ingest
from processes.seed_database import seed_database

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('-rp', '--run-preprocess', action='store_true', help='Runs preprocessor and save report')
parser.add_argument('-rc', '--run-call', type=str, help='runs model with a string input')
parser.add_argument('-rt', '--run-test', action='store_true', help='runs model with a string input')
parser.add_argument('-rs', '--run-service', action='store_true', help='runs the api')
parser.add_argument('-ri', '--run-ingestion', action='store_true', help='runs the ingestion')
parser.add_argument('-rsd', '--run-seeding-database', action='store_true', help='Populates the database with data from the dataset')
args = parser.parse_args()

def main():
    if args.run_preprocess & (args.run_call is not None) & args.run_test & args.run_service & args.run_ingestion & args.run_seeding_database:
        raise ValueError('cant have args together')

    if args.run_preprocess:
        pass

    if args.run_test:
        run_test()

    if args.run_call:
        model = MnliModel()
        result = model.query(args.run_call)
        print(result)

    # if args.run_service:
    #     run_fast()

    if args.run_ingestion:
        ingest()

    if args.run_seeding_database:
        seed_database()


if __name__ == "__main__":
    main()
