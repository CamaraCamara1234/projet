from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from deepface import DeepFace
import os
from django.conf import settings
from PIL import Image
import shutil
from .services.liveness_face_services.test import test
import cv2
import numpy as np
from .services.face_reg_service import *
import base64
import uuid
import json
import binascii


DLIB_THRESHOLD = 0.07
ARC_THRESHOLD = 0.68

# Vue d'enregistrement
LOCK_FILE = os.path.join(settings.BASE_DIR, 'register_user.lock')


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


def verify_faces_advanced(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

    try:
        # Chemin de l'image de référence
        img1_path = os.path.join(
            settings.MEDIA_ROOT, 'extracted_regions', 'photo.png')
        if not os.path.exists(img1_path):
            return JsonResponse({'error': 'Image de référence introuvable'}, status=404)

        img2 = request.FILES.get('image')
        if not img2:
            return JsonResponse({'error': 'Une image est requise'}, status=400)

        extracted_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_regions')
        os.makedirs(extracted_dir, exist_ok=True)
        img2_path = os.path.join(extracted_dir, 'photo_capture.png')

        try:
            img = Image.open(img2)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(img2_path, 'PNG')
        except Exception as img_error:
            return JsonResponse({'error': f'Erreur de traitement de l\'image: {str(img_error)}'}, status=400)

        # Étape 2: Détection de vivacité (liveness)
        try:
            analysis = DeepFace.analyze(
                img_path=img2_path,
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=True
            )

            model_dir = os.path.join(
                settings.BASE_DIR, "facial_recognition/face_models/anti_spoof_models")
            # Tu peux ici ajouter d'autres critères pour détecter les tentatives de triche
            # Placeholder – remplace par une vraie vérification si possible
            # Convertir l’image PIL en image OpenCV
            opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            is_live, confidences = test(
                image=opencv_image, model_dir=model_dir, device_id=0)
            print("liveness detection ====> : ",
                  is_live, " confidence : ", confidences)
        except Exception as live_error:
            return JsonResponse({'error': f'Erreur liveness: {str(live_error)}'}, status=400)

        # Étape 3: Détection de grimaces
        detected_emotion = analysis[0]['dominant_emotion']
        # définis ici ce que tu considères comme "grimace"
        grimaces = ['angry', 'disgust', 'fear', 'surprise', "sad"]
        has_grimace = detected_emotion in grimaces
        print("Emotion detectée : ", detected_emotion)
        print("Has grimace : ", has_grimace)

        if has_grimace:
            return JsonResponse({'error': f'Photo refusée: Grimace détectée ({detected_emotion}).'}, status=400)

        if not is_live:
            return JsonResponse({'error': 'Liveness check failed (visage non-vivant ou artificiel).'}, status=400)

        # Étape 4: Similarité
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="Dlib",
            detector_backend="opencv",
            enforce_detection=False,
            align=True
        )

        distance = result['distance']
        verified = distance <= DLIB_THRESHOLD
        similarity_percent = calculate_similarity(distance)

        if not verified:
            return JsonResponse({'error': 'Visages non similaires.', 'confidence': similarity_percent}, status=401)

        return JsonResponse({
            'verified': True,
            'distance': float(distance),
            'threshold': DLIB_THRESHOLD,
            'confidence': similarity_percent,
            'dominant_emotion': detected_emotion,
            'model': 'Dlib',
            'uploaded_image': settings.MEDIA_URL + "extracted_regions/photo_capture.png"
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def register_face(request):
    if request.method == 'POST':
        try:
            # Récupération des données
            if request.FILES:
                username = request.POST.get('username')
                img_file = request.FILES.get('photo')
            else:
                data = json.loads(request.body.decode('utf-8'))
                username = data.get('username')
                img_data = data.get('photo')

            if not username or (not img_file and not img_data):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Champs username et photo requis'
                }, status=400)

            # Traitement de l'image
            if img_file:
                ext = img_file.name.split('.')[-1].lower()
                img_bytes = img_file.read()
            else:
                header, base64_str = img_data.split(';base64,')
                ext = header.split('/')[-1].lower()
                img_bytes = base64.b64decode(base64_str)

            # Sauvegarde
            user_dir = os.path.join(settings.MEDIA_ROOT, "data", username)
            os.makedirs(user_dir, exist_ok=True)
            filename = f"{uuid.uuid4()}.{ext}"
            filepath = os.path.join(user_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(img_bytes)

            # Traitement du visage
            preprocessor = FacePreprocessor()
            face = preprocessor.process(filepath)
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(user_dir, processed_filename)
            cv2.imwrite(processed_path, cv2.cvtColor(
                (face * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

            # Entraînement incrémental
            data_manager = FaceDataManager()
            trained = data_manager.incremental_train(username)

            return JsonResponse({
                'status': 'success',
                'filename': processed_filename,
                'trained': trained,
                'message': 'Entraînement réussi' if trained else 'Entraînement différé (plus d\'images nécessaires)'
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


# ----------------------------------------------------------
# ENDPOINT DE VERIFICATION
# ----------------------------------------------------------
@csrf_exempt
def verify_face1(request):
    if request.method == 'POST':
        try:
            # 1. Validation des entrées
            if 'username' not in request.POST or 'photo' not in request.FILES:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Les champs username et photo sont requis'
                }, status=400)

            # 2. Sauvegarde temporaire
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f'{uuid.uuid4()}.jpg')

            with open(temp_path, 'wb+') as f:
                for chunk in request.FILES['photo'].chunks():
                    f.write(chunk)

            # 3. Vérification faciale
            verifier = FaceVerifier()
            result = verifier.verify_face(temp_path, request.POST['username'])

            # 4. Conversion des types NumPy avant sérialisation
            serializable_result = {
                'status': str(result['status']),
                'verified': bool(result['verified']),  # Conversion explicite
                'confidence': float(result['confidence']),
                'threshold': float(result['threshold']),
                'comparisons': int(result['comparisons']),
                'best_match': str(result.get('best_match', ''))
            }

            # 5. Nettoyage
            os.remove(temp_path)

            return JsonResponse(serializable_result)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            success, message = full_training()
            return JsonResponse({
                'status': 'success' if success else 'error',
                'message': message
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


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
