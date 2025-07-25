import pandas as pd
from transformers import pipeline
import torch
from models.facebook_bart_large_mnli import parse_results
from models.model import Model
from schemas.pydantic.model_response import AiResponse, ZeroShotClassificationResult

class MnliModel(Model):
    def __init__(self, target):
        self.classifier_ = self.get_pipeline()

        # TODO change to enums
        if target == 'priority':
            self.candidate_labels = ['high','medium','low'] # TODO check to fix
        elif target == 'queue':
            self.candidate_labels = ['technical', 'general', 'billing']

    def query(self, text_input) -> AiResponse:
        result = ZeroShotClassificationResult(**self.classifier(
            text_input,
            candidate_labels=self.candidate_labels
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
            text_inputs.tolist(),
            candidate_labels=self.candidate_labels
        )
        return pd.DataFrame(classifier_results)