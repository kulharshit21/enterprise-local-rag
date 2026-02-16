"""
Image extraction from PDF documents.
Extracts embedded images and saves them with metadata.
"""

import os
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass, field

from config import settings


@dataclass
class ExtractedImage:
    """Represents an extracted image with metadata."""

    image_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    source_doc: str = ""
    page_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageExtractor:
    """Extract images from PDF documents."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or settings.IMAGES_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_from_pdf(
        self,
        pdf_path: str,
        doc_id: str,
        role_access: List[str] = None,
    ) -> List[ExtractedImage]:
        """
        Extract all images from a PDF file.

        Returns list of ExtractedImage objects with saved file paths.
        """
        import pdfplumber
        from PIL import Image
        import io

        role_access = role_access or ["viewer"]
        extracted = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    images = page.images
                    for img_idx, img_info in enumerate(images):
                        try:
                            # Extract image bounding box
                            x0 = img_info.get("x0", 0)
                            y0 = img_info.get("top", 0)
                            x1 = img_info.get("x1", page.width)
                            y1 = img_info.get("bottom", page.height)

                            # Crop the page to the image area
                            cropped = page.within_bbox((x0, y0, x1, y1))
                            page_image = cropped.to_image(resolution=150)

                            # Save image
                            image_id = str(uuid.uuid4())
                            filename = f"{image_id}.png"
                            filepath = os.path.join(self.output_dir, filename)
                            page_image.save(filepath)

                            extracted_img = ExtractedImage(
                                image_id=image_id,
                                file_path=filepath,
                                source_doc=pdf_path,
                                page_number=page_num + 1,
                                metadata={
                                    "doc_id": doc_id,
                                    "role_access": role_access,
                                    "page": page_num + 1,
                                    "image_index": img_idx,
                                    "bbox": [x0, y0, x1, y1],
                                },
                            )
                            extracted.append(extracted_img)

                        except Exception as e:
                            print(f"Warning: Failed to extract image {img_idx} from page {page_num}: {e}")
                            continue

        except Exception as e:
            print(f"Warning: Failed to process PDF for images: {e}")

        return extracted
