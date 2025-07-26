from typing import List
from loguru import logger
import pandas as pd
import chromadb

from transformers import AutoTokenizer, AutoModel
import torch

from db import get_session_maker_and_engine
from db.dataset import get_dataset
from models.model import Model
from schemas.pydantic.model_response import AiResponse

class LongFormer():
    def __init__(self):
        self.model_name = 'allenai/longformer-base-4096'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.max_length = 4096

        logger.info(f"Init {self.model_name} {self.device}")


    def encode(self, texts: List[str]):

        embeddings = []

        for text in texts:
            if len(text) > self.max_length:
                logger.warning("text truncated")

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state

            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embedding = (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)
            embeddings.append(embedding.squeeze(0))

        return torch.stack(embeddings)

class Rag(Model):
    def __init__(self, target=None, name="rag-store", persistent=False, chroma_file_name=None):
        self.target = target

        if not chroma_file_name:
            chroma_file_name = 'default'

        if persistent:
            self.path_ = f'./chroma/{chroma_file_name}/'
            logger.info(f'Loading Chroma from {self.path_}')
            self.chroma_client = chromadb.PersistentClient(path = self.path_)
        else:
            self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.get_or_create_collection(name=name)
        self.embedder = LongFormer()

    def reset(self):
        logger.warning("Ingesting data will delete the collection if it exists")
        self.chroma_client.delete_collection(name=self.collection.name)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection.name)

    def ingest(self):
        # TODO ask on cli or have args to see if the collection should be deleted or skip the ingestion
        self.reset()

        logger.info("Starting ingesting data")
        session_maker, engine = get_session_maker_and_engine()
        session = session_maker()
        dataset_df = get_dataset(engine, session, set='train')

        logger.info(f"Dataset loaded with {len(dataset_df)} records")

        ids = dataset_df['id'].astype(str).tolist()
        documents = dataset_df['body'].tolist()
        metadatas = dataset_df[['priority', 'queue']].to_dict(orient='records')

        logger.info(f"Starting Processing {len(documents)} documents")

        # batching the documents as it crashes with too many documents (on my pc)
        batch_size = 100
        for i in range(0, len(dataset_df), batch_size):
            batch_documents = documents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_documents)} documents")
            self._add_to_collection(batch_documents, batch_metadatas, batch_ids)

        logger.info("Ingesting data finished")

    def _add_to_collection(self, documents, metadatas, ids):
        embeddings = self.embedder.encode(documents).tolist()
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, text) -> AiResponse:
        if not self.target:
            raise TypeError('target must be specified to query')
        logger.info(f"Loading embedding for text: {text}")
        embeddings= self.embedder.encode([text]).tolist()
        results = self.collection.query(query_embeddings=embeddings[0], n_results=1)
        documents = results['documents'][0][0]
        metadatas = results['metadatas'][0][0]

        if len(documents) == 0:
            predicted = None
        else:
            predicted = metadatas[self.target]

        return AiResponse(
            category=predicted
        )

    def classifier(self, texts):
        if not self.target:
            raise TypeError('target must be specified to classify')

        results__ = []

        logger.info(f"Loading embedding for {len(texts)} texts")
        embeddings = self.embedder.encode(texts).tolist()
        logger.info(f"Loading results for {len(texts)} texts")
        
        results = self.collection.query(query_embeddings=embeddings, n_results=1)

        documents = results['documents']
        metadatas = results['metadatas']

        for text, document, metadata in zip(texts, documents, metadatas):
            if len(document) == 0:
                predicted = None
            else:
                metadata = metadata[0]
                predicted = metadata[self.target]

            results__.append({
                'sequence': text,
                'max_score': None, # TODO find the max distance for all the results to normalize the score/confidence
                'predicted': predicted
            })
        
        return pd.DataFrame(results__)