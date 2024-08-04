import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import chromadb
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd


class CLIPEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.cpu().numpy().tolist()
    #

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.cpu().numpy().tolist()


class ChromaHandler:
    def __init__(
        self, collection_name: str, db_path: str, embedding_func: CLIPEmbeddingFunction
    ):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.results = None
        self.embedding_func = embedding_func
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"Collection already exists. Error: {e}")
            self.collection = self.client.get_collection(name=collection_name)

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadata)

    def query_collection(
        self, query: str, n_results: int, is_text: bool
    ) -> Dict[str, Any]:
        if is_text:
            query_embeddings = self.embedding_func.embed_texts([query])
        else:
            query_embeddings = query
        return self.collection.query(
            query_embeddings=query_embeddings[0],
            n_results=n_results,
            include=["embeddings", "distances", "metadatas"],
        )

    def delete_collection(self):
        self.client.delete_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' deleted successfully.")



def get_paths_from_query_results(results: Dict[str, Any]) -> List[str]:
    return [item["path"] for sublist in results["metadatas"] for item in sublist]


def get_score_from_query_results(results: Dict[str, Any]) -> List[float]:
    return results["distances"][0]


def plot_images(results: Dict[str, Any]):
    image_paths = get_paths_from_query_results(results)
    scores = get_score_from_query_results(results)

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"score = {round(scores[i], 3)}, name = {os.path.basename(img_path)}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def get_query_results_in_figs(
    chromadb_instance: ChromaHandler,
    query: str,
    n_results: int = 3,
    is_text: bool = True,
):
    results = chromadb_instance.query_collection(query, n_results, is_text)
    plot_images(results)


def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    return images



def get_noise_apperance_frequecy_per_n_to_show(
    chromadb_instance: ChromaHandler,
    test_queries: List[str],
    noise_keywords: List[str],
    n_results: List[int],
) -> Dict[int, float]:
    noise_per_n = {}
    for n in n_results:
        noise_per_n[n] = get_noise_apperance_frequecy(
            chromadb_instance, test_queries, noise_keywords, n
        )
    return noise_per_n


def get_noise_apperance_frequecy(
    chromadb_instance: ChromaHandler,
    test_queries: List[str],
    noise_keywords: List[str],
    n_results: int,
) -> float:
    count = 0
    for text_query in test_queries:
        ids = chromadb_instance.query_collection(text_query, n_results, True)["ids"][0]
        for id in ids:
            if noise_keywords in id:
                count += 1
                break
    return count / len(test_queries)
