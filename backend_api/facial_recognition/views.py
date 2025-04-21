from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from deepface import DeepFace
import os
from django.conf import settings
from PIL import Image
import shutil

# ARC_THRESHOLD = 0.07
ARC_THRESHOLD = 0.68


@csrf_exempt
def verify_faces(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

    try:
        # Chemin de l'image de référence
        img1_path = os.path.join(
            settings.MEDIA_ROOT, 'extracted_regions', 'photo.png')

        # Vérifier si l'image de référence existe
        if not os.path.exists(img1_path):
            return JsonResponse({'error': 'Image de référence introuvable'}, status=404)

        # Récupérer l'image envoyée dans la requête
        img2 = request.FILES.get('image')
        if not img2:
            return JsonResponse({'error': 'Une image est requise'}, status=400)

        # Chemin de destination pour l'image uploadée
        extracted_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_regions')
        os.makedirs(extracted_dir, exist_ok=True)
        img2_path = os.path.join(extracted_dir, 'photo_capture.png')

        # Sauvegarder l'image en la convertissant en PNG si nécessaire
        try:
            # Ouvrir l'image avec PIL
            img = Image.open(img2)

            # Convertir en RGB si nécessaire (pour les PNG avec transparence)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Sauvegarder sous le nouveau nom
            img.save(img2_path, 'PNG')
        except Exception as img_error:
            return JsonResponse({'error': f'Erreur de traitement de l\'image: {str(img_error)}'}, status=400)

        # Vérification des visages
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False,
            align=True
        )

        # Préparer la réponse
        distance = result['distance']
        verified = distance <= ARC_THRESHOLD
        similarity_percent = calculate_similarity(distance)

        return JsonResponse({
            'verified': verified,
            'distance': float(result['distance']),
            'threshold': ARC_THRESHOLD,
            'confidence': similarity_percent,
            'model': 'ArcFace',
            'uploaded_image': settings.MEDIA_URL + "extracted_regions/photo_capture.png"
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def clear_media_dirs(request):
    dirs_to_clear = ['preprocessed_imgs', 'extracted_regions']
    cleared = []
    errors = []

    for folder in dirs_to_clear:
        folder_path = os.path.join(settings.MEDIA_ROOT, folder)
        if os.path.exists(folder_path):
            try:
                # Supprimer tous les fichiers dans le dossier
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                cleared.append(folder)
            except Exception as e:
                errors.append(f"Erreur dans {folder}: {str(e)}")
        else:
            errors.append(f"Dossier non trouvé: {folder_path}")

    return JsonResponse({
        "status": "done" if not errors else "partial",
        "cleared_folders": cleared,
        "errors": errors
    })


def calculate_similarity(distance, ARC_THRESHOLD=0.68):
    if distance <= 0.07:  # 0-0.07
        return 100.0
    elif distance <= 0.14:  # 0.07-0.14
        return 90.0
    elif distance <= 0.21:  # 0.14-0.21
        return 80.0
    elif distance <= 0.28:  # 0.21-0.28
        return 70.0
    elif distance <= 0.35:  # 0.28-0.35
        return 60.0
    elif distance <= 0.42:  # 0.35-0.42
        return 50.0
    elif distance <= 0.49:  # 0.42-0.49
        return 40.0
    elif distance <= 0.56:  # 0.49-0.56
        return 30.0
    elif distance <= ARC_THRESHOLD:  # 0.56-0.68
        return 20.0
    else:  # > 0.68
        return 0.0
