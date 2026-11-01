import pytesseract
from PIL import Image, ImageFilter, ImageOps
import shutil
import os

# Set Tesseract path automatically or allow manual override
tesseract_path = os.getenv("TESSERACT_PATH", shutil.which("tesseract"))
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise EnvironmentError("Tesseract executable not found. Please install Tesseract or set TESSERACT_PATH.")

def extract_text_from_image(image: Image) -> str:

    # Image Preprocessing
    # Convert to grayscale
    image = image.convert("L")

    # Resize (upscale by 2x to help OCR if image is small)
    image = image.resize((image.width * 2, image.height * 2))

    # Sharpen the image
    image = image.filter(ImageFilter.SHARPEN)

    # Run OCR
    extracted_text = pytesseract.image_to_string(image)

    # Output the result
    return extracted_text
