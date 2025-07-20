import numpy as np
from transformers import pipeline
from loguru import logger
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pydantic import BaseModel
from typing import List

import matplotlib.pyplot as plt

LIMIT = 100

class ZeroShotClassificationResult(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]
    predicted: str = None

def apply_parse_pre_label(results: List[ZeroShotClassificationResult], confidence=None):
    for idx, result in enumerate(results):
        labels = result['labels']
        max_score = max(result['scores'])
        if confidence is not None:
            if max_score > confidence.iloc[idx]:
                result['predicted'] = 'skip'
                continue

        predicted_label = labels[result['scores'].index(max_score)]
        result['predicted'] = predicted_label

def run_test(cs_tickets_df, is_confident=True):
    device_index = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device_index
    )

    cs_tickets_df = cs_tickets_df.head(LIMIT)
    confidence = cs_tickets_df['confidence']

    targets = cs_tickets_df['target']
    labels = targets.unique()

    text = cs_tickets_df['body'] # TODO concat subject later ['subject', 'body']

    logger.info(f'Targets for one shot learning {labels}')

    result = classifier(
        text.tolist(),
        candidate_labels=labels
    )

    if is_confident:
        apply_parse_pre_label(result, confidence)
        labels = np.append(labels, 'skip')
    else:
        apply_parse_pre_label(result)

    all_predicted = [r['predicted'] for r in result]

    accuracy = accuracy_score(targets, all_predicted)

    cm = confusion_matrix(targets, all_predicted, labels=labels)

    # TODO save confusion matrix to compare modes
    
    n_classes = cm.shape[0]
    cell_size = 1.2
    figsize = (cell_size * n_classes, cell_size * n_classes)

    fig, ax = plt.subplots(figsize=figsize)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap='Blues', values_format='d')
    plt.tight_layout()
    plt.savefig("multi_class_confusion_matrix.png")

    print(all_predicted)
    print(accuracy)
