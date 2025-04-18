# Top of file (import only what you need)
import os
import cv2
import pytesseract
import numpy as np
import re
from paddleocr import PaddleOCR
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
        self.tess_lang = 'ara' if lang == 'ara' else 'fra'
        self._create_directories()

    def _create_directories(self):
        for subdir in ["extracted_regions", "temp"]:
            os.makedirs(os.path.join(settings.BASE_DIR,
                        "media", subdir), exist_ok=True)

    @property
    def paddle_ocr(self):
        if self._paddle_ocr is None:
            self._paddle_ocr = PaddleOCR(
                lang='arabic' if self.lang == 'ara' else self.lang)
        return self._paddle_ocr

    def extract_regions(self, image_path: str) -> List[Tuple[str, str, float]]:
        try:
            results = self.yolo_model.predict(image_path)
            result = results[0]

            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Image non trouvée : {image_path}")

            extracted_dir = os.path.join(
                settings.BASE_DIR, "media", "extracted_regions")
            os.makedirs(extracted_dir, exist_ok=True)

            extracted_regions = []
            list_face = self.get_preprocessed_files()

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = result.names[int(box.cls)]
                confidence = float(box.conf)

                if label in {"pere"} and "new_cin_recto" in list_face:
                    x2 += int(x2 * 0.02)
                if label == "ville_ar" and any(face in list_face for face in ("new_cin_recto", "old_cin_recto")):
                    x2 -= int(x2 * 0.03)
                if label == "ville_ar" and any(face in list_face for face in ("sejour_recto", "sejour_verso")):
                    x2 -= int(x2 * 0.077)
                    y1 += int(y1 * 0.02)

                roi = img[y1:y2, x1:x2]
                h, w = roi.shape[:2]

                if max(h, w) < 150 and label != "sexe":
                    scale = 180 / max(h, w)
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

            # Ajout de l'anglais comme backup pour l'arabe
            lang_tess = "fra" if lang == "fr" else "ara+eng"

            # Pré-traitement spécifique pour l'arabe
            if lang == 'ar':
                # Conversion en niveaux de gris avec meilleur contraste
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                # Dilation légère pour améliorer les caractères arabes
                kernel = np.ones((1, 1), np.uint8)
                gray = cv2.dilate(gray, kernel, iterations=1)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extraction PaddleOCR
            result_paddle = self.paddle_ocr.ocr(image_path)
            text_paddle = " ".join(
                [line[1][0] for line in result_paddle[0]]) if result_paddle and result_paddle[0] else ""

            # Configuration Tesseract optimisée pour l'arabe
            # PSM 6 meilleur pour l'arabe (bloc uniforme de texte)
            psm = "6" if lang == "ar" else "7"
            config = f"--oem 3 --psm {psm} -l {lang_tess}"

            # Ajout de configurations spécifiques pour l'arabe
            if lang == 'ar':
                config += " -c tessedit_char_whitelist=ابتةثجحخدذرزسشصضطظعغفقكلمنهويىئءؤرلاـًٌٍَُِّْ٠١٢٣٤٥٦٧٨٩"

            text_tesseract = pytesseract.image_to_string(
                gray, config=config).strip()

            # Post-traitement spécifique pour l'arabe
            if lang == 'ar':
                # Nettoyage des résultats
                text_tesseract = re.sub(
                    r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]', '', text_tesseract)
                text_paddle = re.sub(
                    r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]', '', text_paddle)

                # Choix du meilleur texte avec priorité à Tesseract pour l'arabe
                best_text = text_tesseract if len(text_tesseract) > len(
                    text_paddle)*0.7 else text_paddle
            else:
                best_text = max([text_paddle, text_tesseract], key=len)

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
                if label != "photo" and label != "photo_portrait":
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
