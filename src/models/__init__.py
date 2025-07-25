import numpy as np
from models.facebook_bart_large_mnli.load_model import MnliModel
from models.model import Model
from processes.preprocess import get_cs_tickets_df
from schemas.pydantic.run_configs import TestConfig
from util.load_configs import load_test_config


import pandas as pd
from loguru import logger


import math

def get_dataset(config):
    cs_tickets_df = get_cs_tickets_df()

    len_of_dataset = cs_tickets_df.shape[0]
    cs_tickets_df = cs_tickets_df.head(math.floor(len_of_dataset*config.test_factor))

    targets = cs_tickets_df['target']
    labels = targets.unique()
    texts = cs_tickets_df['body']

    logger.info(f'Targets for testing: {labels}')
    return cs_tickets_df, targets, labels, texts

def get_model(config: TestConfig) -> tuple[Model, Model]:
    # TODO make model names and targets to enums
    if config.priority_model_name == 'MNLI':
        priority_model = MnliModel(target='priority')
    elif config.priority_model_name == 'RAGLF':
        # priority_model = RagLfModel(target='priority')
        raise ImportError()
    else:
        raise ValueError("Wring model name for priority model")

    if config.queue_model_name == 'MNLI':
        queue_model = MnliModel(target='queue')
    elif config.queue_model_name == 'RAGLF':
        # queue_model = RagLfModel(target='queue')
        raise ImportError()
    else:
        raise ValueError("Wring model name for priority model")

    return priority_model, queue_model


def apply_parse_pre_label(classified_results_df: pd.DataFrame):
    classified_results_df['max_score'] = classified_results_df['scores'].map(lambda scores: max(scores))
    classified_results_df['max_score_arg'] = classified_results_df['scores'].map(lambda scores: np.argmax(scores))
    # classified_results_df['skip'] = classified_results_df['max_score'] < classified_results_df['confidence']
    classified_results_df['predicted'] = classified_results_df.apply(lambda cr_row: cr_row['labels'][cr_row['max_score_arg']], axis=1)

# priority skip to top of the que customer support 
# classified_tickets_df['correct'] = classified_tickets_df['predicted'] == classified_tickets_df['target']
# value_c_correct_w_skips = classified_tickets_df[(classified_tickets_df['correct']) & (~classified_tickets_df['skip'])]['priority'].value_counts().rename("correct")
# value_c_skipped_correct = classified_tickets_df[(classified_tickets_df['correct']) & (classified_tickets_df['skip'])]['priority'].value_counts().rename("skipped correct")
# value_c_skipped_incorrect = classified_tickets_df[(~classified_tickets_df['correct']) & (classified_tickets_df['skip'])]['priority'].value_counts().rename("skipped incorrect")
# value_c_incorrect_w_skip = classified_tickets_df[(~classified_tickets_df['correct']) & (~classified_tickets_df['skip'])]['priority'].value_counts().rename("incorrect not skipped")
# value_c_correct = classified_tickets_df[classified_tickets_df['predicted'] == classified_tickets_df['target']]['priority'].value_counts().rename('correct no skip')
# values = pd.concat([value_c_all, value_c_correct, value_c_skipped_correct, value_c_skipped_incorrect, value_c_correct_w_skips, value_c_incorrect_w_skip], axis=1)
# values.fillna(0, inplace=True)
# values['correct skip (%)'] = ((values['skipped incorrect']/values['all'])*100).round(2)
# values['correct (%)'] = ((values['correct']/values['all'])*100).round(2)
# values['correct precision'] = (values['correct']/(values['incorrect not skipped']+values['correct'])).round(2)
# values['unless skip (%)'] = ((values['skipped correct']/values['all'])*100).round(2)
# values.to_csv('priority_report.csv')

def save_priority_report(classified_tickets_df):
    print(classified_tickets_df)
    exit()
    # values.to_csv('priority_report.csv')

def run_test():
    config: TestConfig = load_test_config()
    cs_tickets_df, targets, labels, texts = get_dataset(config)
    priority_model, queue_model = get_model(config)

    # priority_results_df = priority_model.classifier(texts)
    # queue_results_df = queue_model.classifier(texts)
    # priority_results_df.to_csv('./priority_results_df.csv')
    # queue_results_df.to_csv('./queue_results_df.csv')

    priority_results_df = pd.read_csv('./priority_results_df.csv')
    queue_results_df = pd.read_csv('./queue_results_df.csv')

    # classified_tickets_df_ = pd.concat([cs_tickets_df, queue_results_df, priority_results_df], axis=1)
    # classified_tickets_df = classified_tickets_df_.loc[:, ~classified_tickets_df_.columns.duplicated()]
    apply_parse_pre_label(priority_results_df)
    apply_parse_pre_label(queue_results_df)

    save_priority_report(priority_results_df)

    # predicted = classified_tickets_df['predicted']

    # targets_filtered = targets[~classified_tickets_df['skip']]
    # predicted_filtered = predicted[~classified_tickets_df['skip']]

    # # Saving reports of how the model labels are doing
    # save_report(targets, predicted, labels, append_file_name="")
    # value_count_skips_target = (classified_tickets_df[['target', 'skip']].value_counts().xs(True, level=1)/len(predicted)).rename('skip factor')
    # save_report(targets_filtered, predicted_filtered, labels, value_count_skips_target, append_file_name="_filtered")

    # # Plotting Confusion Matrix
    # plot_confusion_matrix(targets, predicted, labels)
    # plot_confusion_matrix(targets_filtered, predicted_filtered, labels, append_file_name="_filtered")

    # predicted_skipped_marked = predicted.copy()
    # predicted_skipped_marked[classified_tickets_df['skip']] = 'skip'
    # labels_w_skip = predicted_skipped_marked.unique()

    # plot_confusion_matrix(targets, predicted_skipped_marked, labels_w_skip, append_file_name="_filtered_with_skipped")