from django.http import JsonResponse
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import logging
from .services.extraction_service import ExtractZonesTexts
from .services.detection_service import DetectionService
from camel_tools.tokenizers.word import simple_word_tokenize
from datetime import datetime
import re
import time

logger = logging.getLogger(__name__)

DOC_TYPES = {
    'new_cin_recto': 'cin_recto',
    'new_cin_verso': 'cin_verso',
    'old_cin_recto': 'cin_recto',
    'old_cin_verso': 'cin_verso',
    'sejour_recto': 'cin_recto',
    'sejour_verso': 'sejour_verso'
}


@csrf_exempt
def extract_regions_view(request):
    if request.method != 'POST':
        return JsonResponse({
            "status": "error",
            "message": "Méthode non autorisée"
        }, status=405)

    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse({
            "status": "error",
            "message": "Aucune image fournie"
        }, status=400)

    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, image_file.name)

    try:
        # Sauvegarde temporaire du fichier
        with open(temp_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Extraction des régions
        regions = DetectionService.extract_and_save_regions(temp_path)
        logger.info(f"Résultats de l'extraction : {regions}")

        if not regions:
            return JsonResponse({
                "status": "error",
                "message": "Aucune région détectée"
            }, status=400)

        first_region = regions[0]
        if first_region.get('status') == 'rejected':
            return JsonResponse({
                "status": "rejected",
                "message": first_region.get('message', 'Erreur de validation'),
                "details": first_region
            }, status=400)

        list_files = get_preprocessed_files()
        if len(list_files) == 2:
            return handle_ocr_processing(list_files)

        return JsonResponse({
            "status": "success",
            "regions": regions,
            "count": len(regions)
        })

    except Exception as e:
        logger.error(f"Erreur lors du traitement : {str(e)}")
        return JsonResponse({
            "status": "error",
            "message": f"Erreur de traitement : {str(e)}"
        }, status=500)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@csrf_exempt
def extract_regions_dual_view(request):
    if request.method != 'POST':
        return JsonResponse({
            "status": "error",
            "message": "Méthode non autorisée"
        }, status=405)

    image1 = request.FILES.get('image1')
    image2 = request.FILES.get('image2')

    if not image1 or not image2:
        return JsonResponse({
            "status": "error",
            "message": "Deux images sont requises"
        }, status=400)

    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_path_1 = os.path.join(temp_dir, image1.name)
    temp_path_2 = os.path.join(temp_dir, image2.name)

    try:
        # Sauvegarde temporaire des fichiers
        for image_file, temp_path in [(image1, temp_path_1), (image2, temp_path_2)]:
            with open(temp_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

        # Extraction des régions pour les deux images
        regions1 = DetectionService.extract_and_save_regions(
            temp_path_1, skip_validation=2)
        regions2 = DetectionService.extract_and_save_regions(
            temp_path_2, skip_validation=2)

        if not regions1 or not regions2:
            return JsonResponse({
                "status": "error",
                "message": "Une ou les deux images n'ont pas permis de détecter des régions"
            }, status=400)

        # Vérification des statuts
        for regions in [regions1, regions2]:
            first = regions[0]
            if first.get('status') == 'rejected':
                return JsonResponse({
                    "status": "rejected",
                    "message": first.get('message', 'Erreur de validation'),
                    "details": first
                }, status=400)

        list_files = get_preprocessed_files()
        if len(list_files) >= 2:
            return handle_ocr_processing(list_files)

        return JsonResponse({
            "status": "success",
            "regions_image1": regions1,
            "regions_image2": regions2
        })

    except Exception as e:
        logger.error(f"Erreur lors du traitement double image : {str(e)}")
        return JsonResponse({
            "status": "error",
            "message": f"Erreur de traitement : {str(e)}"
        }, status=500)

    finally:
        # Suppression des fichiers temporaires
        for temp_path in [temp_path_1, temp_path_2]:
            if os.path.exists(temp_path):
                os.remove(temp_path)


@csrf_exempt
def extract_regions_front_view(request):
    if request.method != 'POST':
        return JsonResponse({
            "status": "error",
            "message": "Méthode non autorisée"
        }, status=405)

    image1 = request.FILES.get('image1')

    if not image1:
        return JsonResponse({
            "status": "error",
            "message": "Une image est requise"
        }, status=400)

    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_path_1 = os.path.join(temp_dir, image1.name)

    try:
        # Sauvegarde temporaire des fichiers
        for image_file, temp_path in [(image1, temp_path_1)]:
            with open(temp_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

        # Extraction des régions pour les deux images
        regions1 = DetectionService.extract_and_save_regions(
            temp_path_1, skip_validation=2)

        if not regions1:
            return JsonResponse({
                "status": "error",
                "message": "l'images n'a pas permis de détecter des régions"
            }, status=400)

        # Vérification des statuts
        for regions in [regions1]:
            first = regions[0]
            if first.get('status') == 'rejected':
                return JsonResponse({
                    "status": "rejected",
                    "message": first.get('message', 'Erreur de validation'),
                    "details": first
                }, status=400)

        list_files = get_preprocessed_files()
        if len(list_files) == 1:
            return handle_ocr_processing(list_files)

        return JsonResponse({
            "status": "success",
            "regions_image1": regions1
        })

    except Exception as e:
        logger.error(f"Erreur lors du traitement double image : {str(e)}")
        return JsonResponse({
            "status": "error",
            "message": f"Erreur de traitement : {str(e)}"
        }, status=500)

    finally:
        # Suppression des fichiers temporaires
        for temp_path in [temp_path_1]:
            if os.path.exists(temp_path):
                os.remove(temp_path)


################################ les fonctions ########################################

def handle_ocr_processing(list_files):
    try:
        t1 = time.time()
        resultats = process_ocr_for_files(list_files)
        list_regions_name = get_regions_files()
        best_results = {}
        mrz_data = []

        for result in resultats:
            if not result.text:
                continue

            try:
                label = map_label(result.label, list_files)
                text = process_text(result, label, list_files)

                if label not in best_results or result.confidence > best_results[label]['confidence']:
                    best_results[label] = {
                        'label': label,
                        'text': text,
                        'confidence': result.confidence
                    }

                if result.label == "code" and len(result.text) >= 10:
                    mrz_data.append(mrz_precessing(result.text))

            except Exception as e:
                logger.warning(f"Erreur traitement {result.label}: {e}")

        t2 = time.time()
        return JsonResponse({
            "status": "success",
            "photo": settings.MEDIA_URL + "extracted_regions/photo.png" if "photo" in list_regions_name else "N/A",
            "photo_portrait": settings.MEDIA_URL + "extracted_regions/photo_portrait.png" if "photo_portrait" in list_regions_name else "N/A",
            "mrz_image": settings.MEDIA_URL + "extracted_regions/code.png" if "code" in list_regions_name else "N/A",
            "cin_recto": settings.MEDIA_URL + f"preprocessed_imgs/{list_files[0]}.jpg",
            "cin_verso": settings.MEDIA_URL + f"preprocessed_imgs/{list_files[1]}.jpg" if len(list_files) >= 2 else "N/A",
            "extracted_data": list(best_results.values()),
            "mrz_data": mrz_data,
            "temps": t2 - t1
        }, content_type='application/json')

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return JsonResponse({
            "status": "error",
            "message": f"OCR échoué: {e}"
        }, status=500)


def map_label(label, list_files):
    # Gestion des labels inversés ou remplacés
    if label == "nom":
        return "prenom"
    elif label == "prenom":
        return "nom"
    elif label == "nom_ar":
        return "prenom_ar"
    elif label == "prenom_ar":
        return "nom_ar"
    elif label == "ville" and any(f in list_files for f in ["sejour_recto", "sejour_verso"]):
        return "nationalite"
    elif label == "ville_ar" and any(f in list_files for f in ["sejour_recto", "sejour_verso"]):
        return "nationalite_ar"
    return label


def process_text(result, label, list_files):
    """Nettoie et formate le texte selon son label."""
    text = (result.text or "").strip()

    if not text:
        return "N/A" if label in ["date_naissance", "date_expiration"] else ""

    # Traitements spécifiques par type de label
    processors = {
        "date_naissance": lambda t: clean_and_format_date(t) if len(t) >= 10 else "N/A",
        "date_expiration": lambda t: clean_and_format_date(t) if len(t) >= 10 else "N/A",
        "mere": lambda t: t.replace("Elde ", "").replace("Etde ", "").replace(",", ""),
        "mere_ar": lambda t: ocr_postprocessing(t).replace("0", "").replace("1", ""),
        "pere_ar": lambda t: ocr_postprocessing(t).replace("0", "").replace("1", ""),
        "ville_ar": advanced_clean,
        "nationalite_ar": advanced_clean,
        "ville": lambda t: nettoyage_texte(t.replace("Nationalité ", "")),
        "nationalite": lambda t: nettoyage_texte(t.replace("Nationalité ", "")),
        "adresse": nettoyage_texte
    }

    if label in processors:
        return processors[label](text)

    # Traitement par défaut pour les autres labels
    if label not in {"code", "num_etat_civil", "adresse_ar", "adresse", "cin", "photo", "photo_portrait"}:
        cleaned_text = text.replace("1", "I").replace("<", "").replace(">", "")
        return advanced_clean(cleaned_text)

    return text


def nettoyage_texte(text):
    words = text.strip().split()

    if words:
        first_word = words[0]
        if (not first_word.isupper() or not first_word.isalnum()) and len(words) > 1:
            words = words[1:]

    return " ".join(words)


def correct_ben_bent(text):
    """
    Corrige les fusions spécifiques tout en préservant les mots composés corrects
    """
    # Traitement prioritaire des cas spécifiques
    replacements = [
        (r'(?<! )بنت(?! )(?=[بتكجحخدذرزسشصضطظعغفقكلمنهويأإآى])', 'بنت '),
        (r'(?<! )ابن(?! )(?=[بتكجحخدذرزسشصضطظعغفقكلمنهويأإآى])', 'ابن '),
        (r'(?<! )بن(?! )(?=[بتكجحخدذرزسشصضطظعغفقكلمنهويأإآى])', 'بن ')
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    return text


def ocr_postprocessing(text):
    # Correction des cas spécifiques en premier
    text = correct_ben_bent(text)

    # Tokenisation intelligente qui préserve "بنت" comme un seul token
    tokens = []
    for token in simple_word_tokenize(text):
        # Recombine "بن"+"ت" en "بنت" si nécessaire
        if len(tokens) > 0 and tokens[-1] == 'بن' and token == 'ت':
            tokens[-1] = 'بنت'
        else:
            tokens.append(token)

    return ' '.join(tokens)


def clean_and_format_date(date_text):
    """
    Nettoie et formate une date pour obtenir le format jj.mm.aaaa

    Args:
        date_text (str): Texte contenant une date à nettoyer

    Returns:
        str: Date formatée (jj.mm.aaaa) ou None si non valide
    """
    try:
        if not date_text:
            return None

        # Normaliser les séparateurs : remplacer virgule, espace, tiret par un point
        date_clean = re.sub(r'[, \-]', '.', date_text)
        # Nettoyage spécifique si nécessaire
        date_clean = date_clean.replace('1 ', '')

        # Supprimer tous les caractères non numériques sauf les points
        cleaned = re.sub(r'[^\d.]', '', date_clean)

        # Extraire les segments de date
        parts = [p for p in cleaned.split('.') if p]

        # Cas classique avec des points
        if len(parts) == 3:
            day, month, year = parts

            # Compléter les segments
            day = day.zfill(2)[:2]
            month = month.zfill(2)[:2]
            year = year[-4:].zfill(4)

            # Valider la date
            datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
            return f"{day}.{month}.{year}"

        # Cas sans séparateurs ("05012027")
        elif len(cleaned) >= 8 and '.' not in cleaned:
            cleaned = cleaned.zfill(8)[:8]
            return f"{cleaned[:2]}.{cleaned[2:4]}.{cleaned[4:8]}"

    except (ValueError, AttributeError):
        pass

    return None


def clean_text(text):
    """
    Nettoie le texte en supprimant les caractères non désirés et en normalisant l'espacement.

    Args:
        text (str): Texte à nettoyer

    Returns:
        str: Texte nettoyé
    """
    # Supprimer les marqueurs de direction Unicode (left-to-right mark, right-to-left mark)
    text = re.sub(r'[\u200e\u200f]', '', text)

    # Supprimer les autres caractères spéciaux non désirés (ajuster selon besoins)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Contrôles ASCII

    # Normaliser les espaces (supprimer espaces multiples, saut de ligne, etc.)
    text = ' '.join(text.split())

    return text.strip()


def advanced_clean(text):
    # Nettoyage de base
    text = clean_text(text)

    # Supprimer les caractères isolés non alphabétiques (sauf pour l'arabe/français)
    text = re.sub(
        r'(?<!\w)[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Zéèêëàâäôöûüç]+\s?', '', text)

    # Corriger les apostrophes/guillemets mal reconnus
    text = text.replace('"', "'").replace(
        '“', "'").replace('”', "'").replace("٠", "")

    return text.strip()


def remove_digits(text):
    return ''.join(c for c in text if not c.isdigit())


def get_preprocessed_files():
    """Liste les fichiers prétraités sans extension"""
    preprocessed_dir = getattr(settings, 'PREPROCESSED_IMGS_DIR',
                               os.path.join(settings.BASE_DIR, 'media', 'preprocessed_imgs'))

    os.makedirs(preprocessed_dir, exist_ok=True)

    return [
        os.path.splitext(f)[0]
        for f in os.listdir(preprocessed_dir)
        if f.lower().endswith('.jpg')
    ]


def get_regions_files():
    """Liste les fichiers prétraités sans extension"""
    preprocessed_dir = getattr(settings, 'REGIONS_IMGS_DIR',
                               os.path.join(settings.BASE_DIR, 'media', 'extracted_regions'))

    os.makedirs(preprocessed_dir, exist_ok=True)

    return [
        os.path.splitext(f)[0]
        for f in os.listdir(preprocessed_dir)
        if f.lower().endswith('.png')
    ]


def process_ocr_for_files(file_list):
    """Traite les fichiers avec OCR et retourne tous les résultats"""
    all_results = []

    for face in file_list:
        try:
            model_path = os.path.join(
                settings.BASE_DIR, f"extraction/extraction_models/{DOC_TYPES[face]}/best.onnx")
            logger.info(f"Modèle utilisé : {model_path}")

            if not os.path.exists(model_path):
                logger.warning(f"Aucun modèle trouvé pour : {face}")
                continue

            extractor = ExtractZonesTexts(
                yolo_model_path=model_path,
                lang='fr'
            )

            image_path = os.path.join(
                settings.BASE_DIR,
                f"media/preprocessed_imgs/{face}.jpg"
            )
            logger.info(f"Traitement de l'image : {image_path}")

            if not os.path.exists(image_path):
                logger.warning(f"Fichier introuvable : {image_path}")
                continue

            results = extractor.process_image(image_path)
            if results:
                all_results.extend(results)

        except KeyError as e:
            logger.error(f"Type de document non reconnu pour {face}: {str(e)}")
        except Exception as e:
            logger.error(
                f"Erreur lors du traitement OCR pour {face}: {str(e)}")
            continue

    return all_results


# def mrz_precessing(mrz_code):
#     print("--"*50)
#     print(mrz_code)
#     print("--"*50)
#     mrz_data = {}
#     list_elts = []
#     current = ""
#     i = 0
#     while i < len(mrz_code):
#         if mrz_code[i] == '<':
#             if current:  # if we have accumulated characters
#                 list_elts.append(current)
#                 current = ""
#             # Skip consecutive '<' characters
#             while i < len(mrz_code) and mrz_code[i] == '<':
#                 i += 1
#             # Start a new segment
#             if i < len(mrz_code):
#                 current = mrz_code[i]
#                 i += 1
#         else:
#             current += mrz_code[i]
#             i += 1
#     if current:  # add the last segment if it exists
#         list_elts.append(current)
#     print("=====> : ", list_elts[2])
#     mrz_data["cin_mrz"] = list_elts[1][1:]
#     mrz_data["date_naiss_mrz"] = list_elts[2][4:6] + \
#         '.'+list_elts[2][2:4]+'.'+list_elts[2][0:2]
#     mrz_data["sexe_mrz"] = list_elts[2][9]
#     mrz_data["date_exp_mrz"] = list_elts[2][13:15] + \
#         '.'+list_elts[2][11:13]+'.'+list_elts[2][9:11]
#     mrz_data["pays"] = list_elts[2][-3:]
#     mrz_data["nom_mrz"] = list_elts[3].split(' ')[1] if len(
#         list_elts[3].split(' ')) > 1 else list_elts[3].split(' ')[0][1:]
#     mrz_data["prenom_mrz"] = ' '.join(list_elts[4:])

#     return mrz_data

def mrz_precessing(mrz_code):
    # Supprimer les retours à la ligne et espaces superflus
    mrz_code = mrz_code.replace('\n', '')
    mrz_data = {}
    list_elts = []
    current = ""
    i = 0
    while i < len(mrz_code):
        if mrz_code[i] == '<':
            if current:  # if we have accumulated characters
                list_elts.append(current)
                current = ""
            # Skip consecutive '<' characters
            while i < len(mrz_code) and mrz_code[i] == '<':
                i += 1
            # Start a new segment
            if i < len(mrz_code):
                current = mrz_code[i]
                i += 1
        else:
            current += mrz_code[i]
            i += 1
    if current:  # add the last segment if it exists
        list_elts.append(current)
    mrz_data["cin_mrz"] = list_elts[1][1:]
    # if int(list_elts[2][0:2]) > 25 and int(list_elts[2][0:2]) <= 99:
    mrz_data["date_naiss_mrz"] = list_elts[2][4:6] + \
        '.'+list_elts[2][2:4]+'.'+list_elts[2][0:2]
    mrz_data["sexe_mrz"] = list_elts[2][7]
    mrz_data["date_exp_mrz"] = list_elts[2][12:14] + \
        '.'+list_elts[2][10:12]+'.'+list_elts[2][8:10]
    mrz_data["pays"] = list_elts[2][-3:]
    mrz_data["nom_mrz"] = list_elts[3].split(' ')[1] if len(
        list_elts[3].split(' ')) > 1 else list_elts[3].split(' ')[0][1:]
    mrz_data["prenom_mrz"] = list_elts[4]

    return mrz_data
