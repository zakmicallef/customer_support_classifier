from api import run_fast
from models.facebook_bart_large_mnli import run_test
from models.facebook_bart_large_mnli.load_model import LoadModel
from processes.preprocess import preprocess, get_cs_tickets_df
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('-rp', '--run-preprocess', action='store_true', help='Runs preprocessor and save report')
parser.add_argument('-rc', '--run-call', type=str, help='runs model with a string input')
parser.add_argument('-rt', '--run-test', action='store_true', help='runs model with a string input')
parser.add_argument('-rs', '--run-service', action='store_true', help='runs the api')
args = parser.parse_args()

def main():
    if args.run_preprocess & (args.run_call is not None) & args.run_test & args.run_service:
        raise ValueError('cant have args together')

    if args.run_preprocess:
        preprocess()

    if args.run_test:
        df = get_cs_tickets_df()
        run_test(df)

    if args.run_call:
        model = LoadModel(['technical', 'general', 'billing'])
        result = model.query(args.run_call)
        print(result)

    # if args.run_service:
    #     run_fast()

if __name__ == "__main__":
    main()
