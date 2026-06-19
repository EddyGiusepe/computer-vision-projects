#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script text_extraction_using_a_small_ocr_model.py
=================================================
Neste script vamos extrair o texto de uma imagem de um
recibo usando um modelo pequeno de OCR. Este exemplo está
baseado no tutorial do CODECUT:
        
RUN
---
uv run text_extraction_using_a_small_ocr_model.py

Install
-------
uv add paddleocr transformers torch torchvision

Link dos modelos
----------------
https://huggingface.co/collections/PaddlePaddle/pp-ocrv6
"""
from pathlib import Path
from urllib.request import urlretrieve

from paddleocr import PaddleOCR

receipt_url = "https://raw.githubusercontent.com/Asprise/receipt-ocr/main/receipt.jpg"
receipt_path = Path("receipt.jpg")

# Download the receipt image locally.
urlretrieve(receipt_url, receipt_path)

# Use tiny OCR models for a lightweight receipt example.
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",  # Locate text regions
    text_recognition_model_name="PP-OCRv6_small_rec",  # Read detected text
    engine="transformers",  # Use the Transformers backend
    use_doc_orientation_classify=False,  # Skip document orientation classification
    use_doc_unwarping=False,  # Skip image unwarping
    use_textline_orientation=True,  # Handle rotated text lines
)

# Run OCR and save visual/text outputs.
results = ocr.predict(str(receipt_path))
#print(results)
for result in results:
    recognized_text = "\n".join(result["rec_texts"])
    print(recognized_text)
