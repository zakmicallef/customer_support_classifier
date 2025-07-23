from typing import List
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings


# name of collection
# input
# target
# collection

from transformers import AutoTokenizer, AutoModel
import torch

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
                last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embedding = (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)
            embeddings.append(embedding.squeeze(0))  # shape [hidden_size]

        return torch.stack(embeddings)  # shape [batch_size, hidden_size]

# class 
# 3. Load embedding model
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = embedder.encode(documents).tolist()

# ...

# query = "What is FastAPI used for?"
# query_embedding = embedder.encode([query]).tolist()[0]

# print(results)

    pass

LIMIT = 100
class Rag():
    def __init__(self, name="rag-demo", persistent=False):
        if persistent:
            self.chroma_client = chromadb.PersistentClient(
                path=f'.',
            )
        else:
            self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=name)

        self.embedder = LongFormer()

    def ingest(self, dataset_df):
        dataset_df = dataset_df.head(LIMIT)

        documents = dataset_df['body'].tolist()
        # TODO replace with db index
        ids = [f"doc{i}" for i in range(len(dataset_df))]
        dataset_df['join'] = dataset_df[['priority', 'target']].to_dict(orient='records')
        metas = dataset_df['join'].tolist()

        logger.info("Starting Processing embeddings")
        embeddings = self.embedder.encode(documents).tolist()

        # TODO add priority and service as metas
        logger.info("Starting adding to Collection")
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metas, ids=ids)

    def query(self, texts): # List[AiResponse]
        logger.info(f"Loading embedding for {len(texts)} texts")
        embeddings = self.embedder.encode(texts).tolist()
        logger.info(f"Loading results for {len(texts)} texts")
        results = self.collection.query(query_embeddings=embeddings, n_results=1)
        print(results)



{
    'ids': [['doc55']], 
    'embeddings': None,
    'documents': [
        ['Our project management SaaS application is encountering functionality problems across multiple devices. It appears recent updates or integration conflicts might be the cause. We have already tried clearing the cache, reinstalling the application, and checking for updates, but the issue still persists. We need assistance to resolve this problem.']
    ],
    'uris': None,
    'included': ['metadatas', 'documents', 'distances'],
    'data': None,
    'metadatas': [
        [
            {'target': 'technical', 'priority': 'high'}
        ]
        ], 'distances': [[2.589689016342163]
    ]
}




# retrieved_doc = results['documents'][0][0]
# # 7. Generate answer using a language model
# generator = pipeline("text2text-generation", model="google/flan-t5-base")
# context = f"Context: {retrieved_doc}\nQuestion: {query}"
# answer = generator(context, max_new_tokens=100)[0]['generated_text']
# print("Question:", query)
# print("Answer:", answer)
