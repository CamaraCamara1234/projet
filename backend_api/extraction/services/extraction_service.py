# Top of file (import only what you need)
import os
import cv2
import pytesseract
import numpy as np
import re
from paddleocr import PaddleOCR
import easyocr
from ultralytics import YOLO
from typing import List, Tuple
from dataclasses import dataclass
import logging
from django.conf import settings

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    label: str
    image_path: str
    text: str
    confidence: float
    ocr_engine: str


class ExtractZonesTexts:
    def __init__(self, yolo_model_path: str, lang: str = 'fr'):
        self.lang = lang
        self.yolo_model = YOLO(yolo_model_path)
        self._paddle_ocr = None
        self._easy_ocr = None
        self.tess_lang = 'ara' if lang == 'ara' else 'fra'
        self._create_directories()

    def _create_directories(self):
        for subdir in ["corrected_imgs", "extracted_regions", "temp"]:
            os.makedirs(os.path.join(settings.BASE_DIR,
                        "extraction", subdir), exist_ok=True)

    @property
    def paddle_ocr(self):
        if self._paddle_ocr is None:
            self._paddle_ocr = PaddleOCR(
                lang='arabic' if self.lang == 'ara' else self.lang)
        return self._paddle_ocr

    @property
    def easy_ocr(self):
        if self._easy_ocr is None:
            self._easy_ocr = easyocr.Reader(
                ['ar'] if self.lang == 'ara' else ['fr'], gpu=False)
        return self._easy_ocr

    def extract_regions(self, image_path: str) -> List[Tuple[str, str, float]]:
        try:
            results = self.yolo_model.predict(image_path)
            result = results[0]

            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Image non trouvée : {image_path}")

            extracted_dir = os.path.join(
                settings.BASE_DIR, "extraction", "extracted_regions")
            os.makedirs(extracted_dir, exist_ok=True)

            extracted_regions = []
            list_face = self.get_preprocessed_files()

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = result.names[int(box.cls)]
                confidence = float(box.conf)

                if label in {"adresse_ar"} and "new_cin_recto" in list_face:
                    x2 += int(x2 * 0.04)

                roi = img[y1:y2, x1:x2]
                h, w = roi.shape[:2]

                if max(h, w) < 250:
                    scale = 250 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    roi = cv2.resize(roi, (new_w, new_h),
                                     interpolation=cv2.INTER_LANCZOS4)

                output_path = os.path.join(extracted_dir, f"{label}.png")

                # Enregistrement avec qualité maximale (JPEG)
                cv2.imwrite(output_path, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])

                extracted_regions.append((label, output_path, confidence))

            return extracted_regions

        except Exception as e:
            logger.error(f"[Extraction Régions] Erreur : {str(e)}")
            raise

    def extract_text(self, image_path: str, lang='fr') -> ExtractionResult:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image non trouvée : {image_path}")

            lang_tess = "fra" if lang == "fr" else "ara"
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            result_paddle = self.paddle_ocr.ocr(image_path)
            text_paddle = " ".join(
                [line[1][0] for line in result_paddle[0]]) if result_paddle and result_paddle[0] else ""

            text_tesseract = pytesseract.image_to_string(
                gray, config=f"--oem 3 --psm 7 -l {lang_tess}").strip()

            result_easy = self.easy_ocr.readtext(image_path, detail=0)
            text_easy = " ".join(result_easy).strip()

            if lang == 'ar' and text_tesseract:
                best_text = text_tesseract
            else:
                best_text = max(
                    [text_paddle, text_tesseract, text_easy], key=len)

            return ExtractionResult(
                label=os.path.basename(image_path),
                image_path=image_path,
                text=best_text,
                confidence=1.0,
                ocr_engine="combined"
            )
        except Exception as e:
            logger.error(f"[OCR] Erreur d'extraction du texte : {str(e)}")
            raise

    def get_preprocessed_files(self):
        """Liste les fichiers prétraités sans extension"""
        preprocessed_dir = getattr(settings, 'PREPROCESSED_IMGS_DIR',
                                   os.path.join(settings.BASE_DIR, 'extraction', 'preprocessed_imgs'))

        os.makedirs(preprocessed_dir, exist_ok=True)

        return [
            os.path.splitext(f)[0]
            for f in os.listdir(preprocessed_dir)
            if f.lower().endswith('.jpg')
        ]

    def process_image(self, image_path: str) -> List[ExtractionResult]:
        try:
            regions = self.extract_regions(image_path)
            results = []
            for label, region_path, confidence in regions:
                try:
                    lang = "ar" if "_ar" in label else "fr"
                    result = self.extract_text(region_path, lang=lang)
                    result.label = label
                    result.confidence = confidence
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        f"[Process Image] Échec traitement région {region_path}: {str(e)}")
            return results
        except Exception as e:
            logger.error(
                f"[Process Image] Échec traitement image {image_path}: {str(e)}")
            raise
