from loguru import logger
from db import get_session_maker_and_engine
from db.dataset import get_dataset
from models import get_model
from models.charts_reports import save_report, apply_charts
from schemas.pydantic.run_configs import TestConfig
from util.load_configs import load_test_config


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

def run_test():
    logger.info("Starting ingesting data")

    config: TestConfig = load_test_config()
    session_maker, engine = get_session_maker_and_engine()
    session = session_maker()
    dataset_df = get_dataset(engine, session, set='test')

    # TODO Re implement this for the whole dataset
    # apply_charts(dataset_df, file_name=config.test_name)

    texts = dataset_df['body']
    priority_labels = dataset_df['priority'].unique().tolist()
    queue_labels = dataset_df['queue'].unique().tolist()
    priority_model, queue_model = get_model(config)

    priority_results_df = priority_model.classifier(texts)
    queue_results_df = queue_model.classifier(texts)

    save_report(
        dataset_df['queue'],
        queue_results_df['predicted'],
        queue_labels,
        skipped=None,
        append_file_name="_queue",
        file_name=config.test_name
    )
    save_report(
        dataset_df['priority'],
        priority_results_df['predicted'],
        priority_labels,
        skipped=None,
        append_file_name="_priority",
        file_name=config.test_name
    )

    queue_targets, queue_predicted, all_labels = get_target_and_predicted_results(queue_results_df, priority_results_df, dataset_df, config)

    save_report(
        queue_targets,
        queue_predicted,
        all_labels,
        skipped=None,
        append_file_name="_all",
        file_name=config.test_name
    )