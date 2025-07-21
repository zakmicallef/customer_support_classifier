import numpy as np
import pandas as pd
from transformers import pipeline
from loguru import logger
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from typing import List
import matplotlib.pyplot as plt

from schemas.pydantic.model_response import AiResponse, ZeroShotClassificationResult

LIMIT = 100

def parse_results(results: ZeroShotClassificationResult) -> AiResponse:
    print(results)
    predicted = results.labels[results.scores.index(max(results.scores))]
    score = results.scores[results.scores.index(max(results.scores))]
    confidence = score/sum(results.scores)
    return AiResponse(category=predicted, confidence=confidence, summary='')

def apply_parse_pre_label(classified_results_df: pd.DataFrame):
    classified_results_df['max_score'] = classified_results_df['scores'].map(lambda scores: max(scores))
    classified_results_df['max_score_arg'] = classified_results_df['scores'].map(lambda scores: np.argmax(scores))
    classified_results_df['skip'] = classified_results_df['max_score'] < classified_results_df['confidence']
    classified_results_df['predicted'] = classified_results_df.apply(lambda cr_row: cr_row['labels'][cr_row['max_score_arg']], axis=1)

def get_pipeline():
    device_index = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device_index
    )

def get_classified_result(text_inputs: pd.Series, candidate_labels):
    classifier = get_pipeline()
    classifier_results = classifier(
        text_inputs.tolist(),
        candidate_labels=candidate_labels
    )
    return pd.DataFrame(classifier_results)

def plot_custom_confusion_matrix(cm, true_labels, pred_labels, title="Confusion Matrix", filename=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(true_labels)))

    ax.set_xticklabels(pred_labels, rotation=45, ha='right')
    ax.set_yticklabels(true_labels)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    plt.tight_layout()

    if filename:
        fig.savefig(filename)

def plot_confusion_matrix(targets, predicted, labels, append_file_name=""):
    labels: List = labels.tolist()
    cm = confusion_matrix(targets, predicted, labels=labels)

    skip_label = 'skip'
    if skip_label in labels:
        remove_idx = labels.index(skip_label)
        new_indices = [labels.index(label) for label in labels if label != skip_label] + [remove_idx]
        reordered_labels = [label for label in labels if label != skip_label] + [skip_label]
        cm = cm[:, new_indices]
        cm = np.delete(cm, remove_idx, axis=0)
        true_labels = [label for i, label in enumerate(labels) if i != remove_idx]
        plot_custom_confusion_matrix(cm, true_labels, reordered_labels, title="Confusion Matrix", filename=f"confusion_matrix{append_file_name}.png")
        return


    # Plot
    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot(ax=ax_cm, cmap='Blues', values_format='d', colorbar=False)
    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(f"confusion_matrix{append_file_name}.png")
    plt.close(fig_cm)

def save_report(targets, predicted, labels, skipped=None, append_file_name=""):
    report = classification_report(
        targets, predicted, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(2)
    if skipped is not None:
        report_df = pd.concat([report_df, skipped], axis=1)
    report_df.to_csv(f"report{append_file_name}.csv")

def save_priority_report(classified_tickets_df):
    value_c_all = classified_tickets_df['priority'].value_counts().rename('all')
    classified_tickets_df['correct'] = classified_tickets_df['predicted'] == classified_tickets_df['target']
    value_c_correct_w_skips = classified_tickets_df[(classified_tickets_df['correct']) & (~classified_tickets_df['skip'])]['priority'].value_counts().rename("correct")
    value_c_skipped_correct = classified_tickets_df[(classified_tickets_df['correct']) & (classified_tickets_df['skip'])]['priority'].value_counts().rename("skipped correct")
    value_c_skipped_incorrect = classified_tickets_df[(~classified_tickets_df['correct']) & (classified_tickets_df['skip'])]['priority'].value_counts().rename("skipped incorrect")
    value_c_incorrect_w_skip = classified_tickets_df[(~classified_tickets_df['correct']) & (~classified_tickets_df['skip'])]['priority'].value_counts().rename("incorrect not skipped")

    value_c_correct = classified_tickets_df[classified_tickets_df['predicted'] == classified_tickets_df['target']]['priority'].value_counts().rename('correct no skip')
    values = pd.concat([value_c_all, value_c_correct, value_c_skipped_correct, value_c_skipped_incorrect, value_c_correct_w_skips, value_c_incorrect_w_skip], axis=1)
    values.fillna(0, inplace=True)

    values['correct skip (%)'] = ((values['skipped incorrect']/values['all'])*100).round(2)
    values['correct (%)'] = ((values['correct']/values['all'])*100).round(2)
    values['correct precision'] = (values['correct']/(values['incorrect not skipped']+values['correct'])).round(2)
    values['unless skip (%)'] = ((values['skipped correct']/values['all'])*100).round(2)
    values.to_csv('priority_report.csv')
   
def run_test(cs_tickets_df: pd.DataFrame):
    cs_tickets_df = cs_tickets_df.head(LIMIT)
    targets = cs_tickets_df['target']
    labels = targets.unique()
    text = cs_tickets_df['body']

    logger.info(f'Targets for zero-shot learning: {labels}')
    classifier_results_df = get_classified_result(text, labels)

    classified_tickets_df = pd.concat([cs_tickets_df, classifier_results_df], axis=1)
    apply_parse_pre_label(classified_tickets_df)
    save_priority_report(classified_tickets_df)

    predicted = classified_tickets_df['predicted']

    targets_filtered = targets[~classified_tickets_df['skip']]
    predicted_filtered = predicted[~classified_tickets_df['skip']]

    # Saving reports of how the model labels are doing
    save_report(targets, predicted, labels, append_file_name="")
    value_count_skips_target = (classified_tickets_df[['target', 'skip']].value_counts().xs(True, level=1)/len(predicted)).rename('skip factor')
    save_report(targets_filtered, predicted_filtered, labels, value_count_skips_target, append_file_name="_filtered")

    # Plotting Confusion Matrix
    plot_confusion_matrix(targets, predicted, labels)
    plot_confusion_matrix(targets_filtered, predicted_filtered, labels, append_file_name="_filtered")

    predicted_skipped_marked = predicted.copy()
    predicted_skipped_marked[classified_tickets_df['skip']] = 'skip'
    labels_w_skip = predicted_skipped_marked.unique()

    plot_confusion_matrix(targets, predicted_skipped_marked, labels_w_skip, append_file_name="_filtered_with_skipped")
