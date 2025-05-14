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
    """Endpoint pour enregistrer un nouveau visage avec augmentations."""
    if request.method == 'POST':
        try:
            # Gestion des différents formats d'entrée
            if request.FILES:
                username = request.POST.get('username')
                img_file = request.FILES.get('photo')
                img = Image.open(img_file).convert('RGB')
            else:
                data = json.loads(request.body.decode('utf-8'))
                username = data.get('username')
                base64_str = data.get('photo').split(',')[1]
                img_bytes = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Validation des entrées
            if not username or img is None:
                return JsonResponse(
                    {'status': 'error', 'message': 'Champs manquants'},
                    status=400
                )

            # Calcul des embeddings
            embeddings = compute_embeddings(img)
            if not embeddings:
                return JsonResponse(
                    {'status': 'error', 'message': 'Visage non détecté'},
                    status=400
                )

            # Sauvegarde des embeddings
            for emb in embeddings:
                save_embedding(username, emb)

            return JsonResponse({
                'status': 'success',
                'message': f'{len(embeddings)} visages enregistrés'
            })

        except Exception as e:
            return JsonResponse(
                {'status': 'error', 'message': str(e)},
                status=500
            )

@csrf_exempt
def verify_face1(request):
    """Endpoint pour vérifier un visage contre les enregistrements existants."""
    if request.method == 'POST':
        try:
            # Récupération des données
            username = request.POST.get('username')
            img_file = request.FILES.get('photo')

            # Validation des entrées
            if not username or not img_file:
                return JsonResponse(
                    {'status': 'error', 'message': 'Champs manquants'},
                    status=400
                )

            # Détection du visage
            img = Image.open(img_file).convert('RGB')
            face = mtcnn(img)
            if face is None:
                return JsonResponse(
                    {'status': 'error', 'message': 'Visage non détecté'},
                    status=400
                )

            # Calcul de l'embedding
            with torch.no_grad():
                input_emb = facenet(face.unsqueeze(0).to(device))
                input_emb = torch.nn.functional.normalize(input_emb, p=2, dim=1)
                input_emb = input_emb.squeeze(0).cpu().numpy()

            # Récupération des embeddings enregistrés
            stored_embeddings = get_user_embeddings(username)
            if len(stored_embeddings) == 0:
                return JsonResponse(
                    {'status': 'error', 'message': 'Aucune donnée pour cet utilisateur'},
                    status=404
                )

            # Calcul de similarité
            similarities = cosine_similarity([input_emb], stored_embeddings)[0]
            best_score = float(np.max(similarities))
            threshold = 0.7  # Plus haut = plus strict

            return JsonResponse({
                'status': 'success',
                'verified': best_score > threshold,
                'similarity_score': best_score,
                'threshold': threshold
            })

        except Exception as e:
            return JsonResponse(
                {'status': 'error', 'message': str(e)},
                status=500
            )

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
