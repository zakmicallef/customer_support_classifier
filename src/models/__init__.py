import os
import uuid
import numpy as np
from sklearn.metrics import classification_report
from models.facebook_bart_large_mnli import MnliModel
from models.rag import Rag
from models.model import Model
from schemas.pydantic.run_configs import TestConfig
from util.load_configs import load_data_config

import pandas as pd

def get_model(config: TestConfig) -> tuple[Model, Model]:
    # TODO make model names and targets to enums
    if config.priority_model_name == 'MNLI':
        priority_model = MnliModel(target='priority')
    elif config.priority_model_name == 'RAGLF':
        raise NotImplementedError('Queue Prediction with Rag not implemented')

        config_rag = load_data_config()
        priority_model = Rag(target='priority', chroma_file_name=config_rag.rag_save_file_name, persistent=True)
    else:
        raise ValueError("Wring model name for priority model")

    if config.queue_model_name == 'MNLI':
        queue_model = MnliModel(target='queue')
    elif config.queue_model_name == 'RAGLF':
        config_rag = load_data_config()
        queue_model = Rag(target='queue', chroma_file_name=config_rag.rag_save_file_name, persistent=True)
    else:
        raise ValueError("Wring model name for priority model")

    return priority_model, queue_model


def save_report(targets, predicted, labels, skipped=None, append_file_name="", file_name=None):
    if not file_name:
        file_location=f"test_results/{uuid.uuid4()}"
    else:
        file_location=f"test_results/{file_name}"
        # TODO ask on cli if test wants to be replaced
    os.makedirs(file_location, exist_ok=True)

    report = classification_report(
        targets, predicted, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(2)
    if skipped is not None:
        report_df = pd.concat([report_df, skipped], axis=1)
    report_df.to_csv(f"{file_location}report{append_file_name}.csv")

def get_target_and_predicted_results(queue_results_df, priority_results_df, cs_tickets_df, config):
    queue_predicted = queue_results_df['predicted']
    queue_targets = cs_tickets_df['queue']
    queue_certainty = queue_results_df['max_score']

    priority_predicted = priority_results_df['predicted']
    priority_predicted_confidence = priority_predicted.map(config.confidence_map)

    confidence_target = cs_tickets_df['priority'].map(config.confidence_map)

    skip_based_on_priorities_predicted = queue_certainty < priority_predicted_confidence
    skip_based_on_priorities_ground_truth = queue_certainty < confidence_target

    queue_predicted[skip_based_on_priorities_predicted] = 'skip'
    queue_targets[skip_based_on_priorities_ground_truth] = 'skip'

    all_labels = queue_targets.unique().tolist()

    return queue_targets, queue_predicted, all_labels

