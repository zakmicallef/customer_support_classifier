import numpy as np
import torch
from models.model import Model
import pandas as pd
from transformers import pipeline
from schemas.pydantic.model_response import AiResponse, ZeroShotClassificationResult


def parse_results(results: ZeroShotClassificationResult) -> AiResponse:
    predicted = results.labels[results.scores.index(max(results.scores))]
    score = results.scores[results.scores.index(max(results.scores))]
    confidence = score/sum(results.scores)
    return AiResponse(category=predicted, confidence=confidence, summary='')

def apply_parse_pre_label(classified_results_df: pd.DataFrame):
    classified_results_df['max_score'] = classified_results_df['scores'].map(lambda scores: max(scores))
    classified_results_df['max_score_arg'] = classified_results_df['scores'].map(lambda scores: np.argmax(scores))
    classified_results_df['predicted'] = classified_results_df.apply(lambda cr_row: cr_row['labels'][cr_row['max_score_arg']], axis=1)
    return classified_results_df[['sequence', 'max_score', 'predicted']]

class MnliModel(Model):
    def __init__(self, target):
        self.classifier_ = self.get_pipeline()

        # TODO change to enums
        if target == 'priority':
            self.candidate_labels = ['high','medium','low'] # TODO check to fix
        elif target == 'queue':
            self.candidate_labels = ['technical', 'general', 'billing']
        else:
            raise ValueError(f"Unknown target: {target}")

    def query(self, text_input) -> AiResponse:
        result = ZeroShotClassificationResult(**self.classifier(
            text_input
        ))
        return parse_results(result)

    def get_pipeline(self):
        device_index = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_index
        )

    def classifier(self, text_inputs):
        classifier_results = self.classifier_(
            [text_inputs],
            candidate_labels=self.candidate_labels
        )
        return parse_results(classifier_results)