"""
Multimodal image embeddings using CLIP.
"""

from typing import List
import numpy as np


class ImageEmbedder:
    """
    Generate image embeddings using CLIP (openai/clip-vit-base-patch32).
    Enables joint text-image retrieval in the vector store.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load CLIP model and processor."""
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @property
    def dimension(self) -> int:
        """CLIP embedding dimension (512 for base model)."""
        return 512

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for a single image file."""
        from PIL import Image
        import torch

        self._load_model()

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        # Normalize
        embedding = outputs[0].numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple images."""
        return [self.embed_image(path) for path in image_paths]

    def embed_text_for_image_search(self, text: str) -> List[float]:
        """
        Embed text using CLIP's text encoder for cross-modal search.
        This enables text queries to find relevant images.
        """
        import torch

        self._load_model()

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)

        embedding = outputs[0].numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
