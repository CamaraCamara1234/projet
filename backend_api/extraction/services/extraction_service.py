# Top of file (import only what you need)
import os
import cv2
import pytesseract
import numpy as np
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
        self._create_directories()

    def _create_directories(self):
        for subdir in ["extracted_regions", "temp"]:
            os.makedirs(os.path.join(settings.BASE_DIR,
                        "media", subdir), exist_ok=True)

    def _basic_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Un prétraitement minimal qui préserve la qualité originale"""
        # Simple conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Légère amélioration de contraste sans distortion
        gray = cv2.normalize(gray, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX)

        return gray
    # def _basic_preprocess(image: np.ndarray) -> np.ndarray:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     contrast = clahe.apply(gray)
    #     kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    #     sharpened = cv2.filter2D(contrast, -1, kernel)
    #     return sharpened

    def _expand_roi(self, roi: np.ndarray, percent: float = 0.15) -> np.ndarray:
        """Agrandit la région d'intérêt de 15% par défaut"""
        h, w = roi.shape[:2]

        # Calculer les nouvelles dimensions (15-20% plus grandes)
        new_h = int(h * (1 + percent))
        new_w = int(w * (1 + percent))

        # Redimensionner avec interpolation de haute qualité
        return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

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

            # D'abord extraire photo, photo_portrait et code sans modification
            special_labels = {"photo", "photo_portrait", "code"}
            special_regions = []

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = result.names[int(box.cls)]
                confidence = float(box.conf)

                # Découpage normal de la région
                roi = img[y1:y2, x1:x2]

                if label in special_labels:
                    # Sauvegarder les zones spéciales sans modification
                    output_path = os.path.join(extracted_dir, f"{label}.png")
                    cv2.imwrite(output_path, roi, [
                                cv2.IMWRITE_JPEG_QUALITY, 100])
                    special_regions.append((label, output_path, confidence))
                    continue

                # Pour les autres zones, appliquer le prétraitement en gris
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

                # Appliquer l'agrandissement de 15-20% après le découpage
                roi_expanded = self._expand_roi(roi_gray, percent=0.30)
                roi_expanded = cv2.convertScaleAbs(
                    roi_expanded, alpha=1.0, beta=0)

                output_path = os.path.join(extracted_dir, f"{label}.png")
                cv2.imwrite(output_path, roi_expanded, [
                            cv2.IMWRITE_JPEG_QUALITY, 100])
                extracted_regions.append((label, output_path, confidence))

            return special_regions + extracted_regions

        except Exception as e:
            logger.error(f"[Extraction Régions] Erreur : {str(e)}")
            raise

    def extract_text(self, image_path: str, lang='fr') -> ExtractionResult:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image non trouvée : {image_path}")

            label = os.path.basename(image_path).replace(".png", "")
            if label == "code":
                # Prétraitement minimal
                processed_image = self._basic_preprocess(image)
                config = f'--psm 11 --oem 3 -c preserve_interword_spaces=1'
            else:
                processed_image = cv2.imread(image_path)
                config = f'--psm 6 --oem 3 -c preserve_interword_spaces=1'

            # Configuration Tesseract
            tess_lang = 'ara' if lang == 'ar' else 'fra'

            # Essayer différents modes PSM si nécessaire
            # for psm in [6, 4]:  # Essayez différents modes de segmentation
            text = pytesseract.image_to_string(
                processed_image,
                lang=tess_lang,
                config=config
            ).strip()

            return ExtractionResult(
                label=os.path.basename(image_path),
                image_path=image_path,
                text=text,
                confidence=1.0,
                ocr_engine="tesseract"
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
                if label not in {"photo", "photo_portrait"}:
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
