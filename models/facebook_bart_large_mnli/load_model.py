from models.facebook_bart_large_mnli import get_pipeline


class LoadModel():
    def __init__(self, candidate_labels):
        self.classifier = get_pipeline()
        self.candidate_labels = candidate_labels

    def query(self, text_input):
        return self.classifier(
            text_input,
            candidate_labels=self.candidate_labels
        )