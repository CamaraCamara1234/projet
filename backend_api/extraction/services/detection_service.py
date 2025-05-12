import cv2
import numpy as np
import os
from typing import List, Dict, Any
from ultralytics import YOLO
from django.conf import settings
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkewCorrector:
    """Classe optimisée pour la correction d'inclinaison des images"""

    @staticmethod
    def _get_rotation_matrix(image: np.ndarray, angle: float) -> np.ndarray:
        """Calcule la matrice de rotation une seule fois"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        return cv2.getRotationMatrix2D(center, angle, 1.0)

    @staticmethod
    def _find_horizontal_lines(edges: np.ndarray) -> List[tuple]:
        """Détecte les lignes horizontales avec des paramètres optimisés"""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=100, minLineLength=30, maxLineGap=10)
        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.hypot(x2 - x1, y2 - y1)

            if -20 < angle < 20:  # Angle range hardcodé pour la performance
                filtered_lines.append((length, angle))

        # Top 5 lignes
        return sorted(filtered_lines, key=lambda x: x[0], reverse=True)[:5]

    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """Corrige l'inclinaison de l'image avec des optimisations"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            filtered_lines = SkewCorrector._find_horizontal_lines(edges)
            if not filtered_lines:
                return image

            angle_median = np.median([angle for _, angle in filtered_lines])

            # Rotation
            M = SkewCorrector._get_rotation_matrix(image, angle_median)
            corrected = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Recadrage optimisé
            gray_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_corr, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                x, y, w, h = cv2.boundingRect(np.vstack(contours))
                corrected = corrected[y:y+h, x:x+w]

            return corrected

        except Exception as e:
            logger.error(
                f"Erreur lors de la correction d'inclinaison: {str(e)}")
            return image


class DetectionService:
    _model = None
    _doc_types = {
        'new_cin_recto': 'new_cin_verso',
        'old_cin_recto': 'old_cin_verso',
        'sejour_recto': 'sejour_verso'
    }

    _dict_classes = {
        0: 'new_cin_recto',
        1: 'new_cin_verso',
        2: 'old_cin_recto',
        3: 'old_cin_verso',
        4: 'sejour_recto',
        5: 'sejour_verso'
    }

    @classmethod
    def get_model(cls) -> YOLO:
        """Cache le modèle pour éviter de le recharger à chaque appel"""
        if cls._model is None:
            model_path = os.path.join(
                settings.BASE_DIR, "extraction/extraction_models/carte_classification/best.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle introuvable : {model_path}")
            cls._model = YOLO(model_path)
        return cls._model

    @classmethod
    def _validate_document_sequence(cls, class_name: str, existing_files: List[str]) -> Dict[str, Any]:
        """Validation optimisée de la séquence de documents"""
        if not existing_files:
            return {"valid": class_name not in cls._doc_types.values(),
                    "message": "La première capture doit être le recto" if class_name in cls._doc_types.values() else ""}

        recto = existing_files[0]
        if class_name in cls._doc_types.values():
            expected_verso = cls._doc_types.get(recto)
            return {
                "valid": expected_verso == class_name,
                "message": "" if expected_verso == class_name else "Ce verso ne correspond pas au recto scanné"
            }

        if recto in cls._doc_types and cls._doc_types[recto] not in existing_files:
            return {"valid": False, "message": "Veuillez scanner le verso avant un nouveau recto"}

        return {"valid": True}

    @classmethod
    def _process_region(cls, region: np.ndarray, scale_factor: float, max_dimension: int) -> np.ndarray:
        """Traitement optimisé de la région (redimensionnement)"""
        if scale_factor != 1.0:
            h, w = region.shape[:2]
            region = cv2.resize(region, (int(w * scale_factor), int(h * scale_factor)),
                                interpolation=cv2.INTER_CUBIC)

        h, w = region.shape[:2]
        if max(h, w) > max_dimension:
            ratio = max_dimension / max(h, w)
            region = cv2.resize(region, (int(w * ratio), int(h * ratio)),
                                interpolation=cv2.INTER_AREA if ratio < 0.5 else cv2.INTER_CUBIC)
        return region

    @classmethod
    def extract_and_save_regions(cls, image_path: str, output_dir: str = "preprocessed_imgs",
                                 scale_factor: float = 1.0, max_dimension: int = 1200,
                                 skip_validation: int = 1) -> List[Dict[str, Any]]:
        """Méthode principale optimisée pour l'extraction des régions"""
        try:
            # Chargement de l'image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(
                    f"Impossible de charger l'image : {image_path}")

            # Préparation du répertoire de sortie
            full_output_dir = os.path.join(
                settings.BASE_DIR, "media", output_dir)
            os.makedirs(full_output_dir, exist_ok=True)

            # Liste des fichiers existants (optimisé)
            existing_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(full_output_dir)
                if f.lower().endswith('.jpg')
            ]

            # Prédiction avec le modèle YOLO
            model = cls.get_model()
            results = model.predict(source=image_path, conf=0.5)
            saved_regions = []

            # Traitement des résultats
            for result in results:
                for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    region = img[y1:y2, x1:x2]
                    class_name = cls._dict_classes[int(cls_id)]

                    # Validation (si activée)
                    if skip_validation == 1:
                        validation = cls._validate_document_sequence(
                            class_name, existing_files)
                        if not validation["valid"]:
                            return [{
                                "path": '',
                                "class": class_name,
                                "original_bbox": box.tolist(),
                                "scaled_size": region.shape[:2],
                                "message": validation["message"],
                                "status": "rejected"
                            }]

                    # Traitement de la région
                    region = cls._process_region(
                        region, scale_factor, max_dimension)

                    # Chemin de sortie et sauvegarde
                    output_path = os.path.join(
                        full_output_dir, f"{class_name}.jpg")
                    corrected_region = SkewCorrector.correct_skew(region)

                    # Sauvegarde optimisée
                    cv2.imwrite(output_path, corrected_region,
                                [cv2.IMWRITE_JPEG_QUALITY, 95] if output_path.lower().endswith(('.jpg', '.jpeg')) else None)

                    saved_regions.append({
                        "path": output_path,
                        "class": class_name,
                        "original_bbox": box.tolist(),
                        "scaled_size": corrected_region.shape[:2],
                        "status": "accepted"
                    })

            return saved_regions

        except Exception as e:
            logger.error(f"Erreur lors du traitement : {str(e)}")
            return [{
                "path": '',
                "class": '',
                "original_bbox": [],
                "scaled_size": [],
                "message": f"Erreur lors du traitement : {str(e)}",
                "status": "error"
            }]
