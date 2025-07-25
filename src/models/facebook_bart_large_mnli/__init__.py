import numpy as np
import pandas as pd
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

def parse_results(results: ZeroShotClassificationResult) -> AiResponse:
    predicted = results.labels[results.scores.index(max(results.scores))]
    score = results.scores[results.scores.index(max(results.scores))]
    confidence = score/sum(results.scores) # TODO ??
    return AiResponse(category=predicted, confidence=confidence, summary='')


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


