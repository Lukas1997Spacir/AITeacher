import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

EMBED_MODEL = "text-embedding-3-small"


def get_embedding(text):
    return client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    ).data[0].embedding


class VectorStore:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.items = []  # metadata + text

    def add_documents(self, chunks, source_name):

        vectors = []

        for c in chunks:
            emb = get_embedding(c["text"])
            vectors.append(emb)

            self.items.append({
                "text": c["text"],
                "chunk_id": c["chunk_id"],
                "source": source_name
            })

        self.index.add(np.array(vectors).astype("float32"))

    def search(self, query, k=5):
        q_emb = np.array([get_embedding(query)]).astype("float32")

        distances, indices = self.index.search(q_emb, k)

        results = []

        for i in indices[0]:
            if i < len(self.items):
                results.append(self.items[i])

        return results
