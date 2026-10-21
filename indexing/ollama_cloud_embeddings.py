
import requests
from langchain_core.embeddings import Embeddings


class OllamaCloudEmbeddings(Embeddings):
    def __init__(
        self,
        model: str,
        api_key: str,
    ):
        self.model = model
        self.api_key = api_key
        self.url = "https://ollama.com/api/embed"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
