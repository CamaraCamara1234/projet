import matplotlib.pyplot as plt
import os
import cv2
from ultralytics import YOLO
from django.conf import settings
from typing import List, Dict, Any
import logging
import numpy as np

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def correct_skew_and_save(image: np.ndarray, save_path: str, angle_range=(-20, 20), top_n=5) -> np.ndarray:
    """
    Redresse une image en détectant les lignes quasi-horizontales et en corrigeant l'inclinaison, puis recadre.

    Args:
        image: image originale (np.ndarray).
        save_path: chemin de sauvegarde de l'image redressée.
        angle_range: plage d'angle en degrés pour considérer les lignes horizontales.
        top_n: nombre de lignes les plus longues à utiliser pour le calcul de l'angle.

    Returns:
        Image redressée et recadrée (np.ndarray).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=100, minLineLength=30, maxLineGap=10)
        angles = []

        corrected = image  # fallback
        rotated = False

        if lines is not None:
            lignes_filtrées = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.hypot(x2 - x1, y2 - y1)

                if angle_range[0] < angle < angle_range[1]:
                    lignes_filtrées.append((length, angle))

            lignes_filtrées = sorted(
                lignes_filtrées, key=lambda x: x[0], reverse=True)[:top_n]

            if lignes_filtrées:
                angles = [angle for _, angle in lignes_filtrées]
                angle_median = np.median(angles)
                print(
                    f"✅ Angle médian détecté pour correction : {angle_median:.2f}°")

                # Rotation inverse
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle_median, 1.0)
                corrected = cv2.warpAffine(
                    image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                rotated = True
            else:
                print("❌ Aucune ligne satisfaisante pour estimer l'inclinaison.")
        else:
            print("❌ Aucune ligne détectée dans l’image.")

        # Recadrage automatique (suppression des bordures vides après rotation)
        if rotated:
            gray_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_corr, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(np.vstack(contours))
                corrected = corrected[y:y+h, x:x+w]

        # Sauvegarde
        if save_path.lower().endswith(".jpg") or save_path.lower().endswith(".jpeg"):
            cv2.imwrite(save_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(save_path, corrected)

    except Exception as e:
        print(f"❌ Erreur : {str(e)}")


# def correct_skew_and_save(image: np.ndarray, save_path: str) -> None:
#     """
#     Corrige l'inclinaison d'une image, améliore légèrement le contraste, applique un défloutage et sauvegarde la meilleure qualité.
#     """
#     try:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(
#             gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#         contours, _ = cv2.findContours(
#             binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contour_image = np.zeros_like(binary)
#         cv2.drawContours(contour_image, contours, -1,
#                          (255), thickness=cv2.FILLED)

#         lines = cv2.HoughLinesP(
#             contour_image, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)
#         angles = []

#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#                 if -10 < angle < 10:
#                     angles.append(angle)

#         if angles:
#             median_angle = np.median(angles)
#             logger.info(
#                 f"Angle d'inclinaison détecté : {median_angle:.2f} degrés")

#             if abs(median_angle) > 1.0:
#                 (h, w) = image.shape[:2]
#                 center = (w // 2, h // 2)
#                 M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#                 image = cv2.warpAffine(
#                     image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#         if save_path.lower().endswith(".jpg") or save_path.lower().endswith(".jpeg"):
#             cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         else:
#             cv2.imwrite(save_path, image)

#         logger.info(f"Image traitée et sauvegardée : {save_path}")

#     except Exception as e:
#         logger.error(
#             f"Erreur lors de la correction et du traitement : {str(e)}")
#         cv2.imwrite(save_path, image)


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
        if cls._model is None:
            model_path = os.path.join(
                settings.BASE_DIR, "extraction/extraction_models/carte_classification/best.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle introuvable : {model_path}")
            cls._model = YOLO(model_path)
        return cls._model

    @classmethod
    def _validate_document_sequence(cls, class_name: str, existing_files: List[str]) -> Dict[str, Any]:
        if not existing_files:
            if class_name in cls._doc_types.values():
                return {"valid": False, "message": "La première capture doit être le recto"}
            return {"valid": True}

        recto = existing_files[0]
        if class_name in cls._doc_types.values():
            expected_verso = cls._doc_types.get(recto)
            if expected_verso != class_name:
                return {"valid": False, "message": "Ce verso ne correspond pas au recto scanné"}
            return {"valid": True}
        elif class_name in existing_files:
            return {"valid": True}

        if recto in cls._doc_types and cls._doc_types[recto] not in existing_files:
            return {"valid": False, "message": "Veuillez scanner le verso avant un nouveau recto"}

        return {"valid": True}

    @classmethod
    def extract_and_save_regions(cls, image_path: str, output_dir: str = "preprocessed_imgs",
                                 scale_factor: float = 1.0, max_dimension: int = 1200, skip_validation: int = 1) -> List[Dict[str, Any]]:
        try:
            model = cls.get_model()
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(
                    f"Impossible de charger l'image : {image_path}")

            full_output_dir = os.path.join(
                settings.BASE_DIR, "media", output_dir)
            os.makedirs(full_output_dir, exist_ok=True)

            existing_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(full_output_dir)
                if f.lower().endswith('.jpg')
            ]

            results = model.predict(source=image_path, conf=0.5)
            saved_regions = []

            for result in results:
                for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    region = img[y1:y2, x1:x2]
                    class_name = cls._dict_classes[int(cls_id)]
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

                    if scale_factor != 1.0:
                        h, w = region.shape[:2]
                        region = cv2.resize(region, (int(
                            w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)

                    h, w = region.shape[:2]
                    if max(h, w) > max_dimension:
                        ratio = max_dimension / max(h, w)
                        region = cv2.resize(region, (int(w * ratio), int(h * ratio)),
                                            interpolation=cv2.INTER_AREA if ratio < 0.5 else cv2.INTER_CUBIC)

                    output_path = os.path.join(
                        full_output_dir, f"{class_name}.jpg")

                    # 🔄 Redressement et sauvegarde directe dans preprocessed_imgs
                    correct_skew_and_save(region, output_path)

                    saved_regions.append({
                        "path": output_path,
                        "class": class_name,
                        "original_bbox": box.tolist(),
                        "scaled_size": region.shape[:2],
                        "status": "accepted"
                    })

            return saved_regions

        except Exception as e:
            return [{
                "path": '',
                "class": '',
                "original_bbox": [],
                "scaled_size": [],
                "message": f"Erreur lors du traitement : {str(e)}",
                "status": "error"
            }]
