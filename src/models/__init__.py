import numpy as np
from models.facebook_bart_large_mnli import MnliModel
from models.rag import Rag
from models.model import Model
from schemas.pydantic.run_configs import TestConfig
from util.load_configs import load_data_config

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



