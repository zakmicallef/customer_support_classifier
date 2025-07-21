from models.facebook_bart_large_mnli import get_pipeline, parse_results
from schemas.model_response import AiResponse, ZeroShotClassificationResult

class LoadModel():
    def __init__(self, candidate_labels):
        self.classifier = get_pipeline()
        self.candidate_labels = candidate_labels

    def query(self, text_input) -> AiResponse:
        result = ZeroShotClassificationResult(**self.classifier(
            text_input,
            candidate_labels=self.candidate_labels
        ))
        return parse_results(result)