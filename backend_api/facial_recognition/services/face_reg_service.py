import os
import uuid
import json
import base64
import io
import torch
import numpy as np
from PIL import Image, ImageEnhance
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Initialisation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=device,
    thresholds=[0.7, 0.8, 0.8]
)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

EMBEDDINGS_DIR = os.path.join(settings.MEDIA_ROOT, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def save_embedding(username, embedding):
    """Sauvegarde l'embedding facial dans le dossier de l'utilisateur."""
    user_dir = os.path.join(EMBEDDINGS_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    np.save(os.path.join(user_dir, f"{uuid.uuid4()}.npy"), embedding)

def get_user_embeddings(username):
    """Récupère tous les embeddings enregistrés pour un utilisateur."""
    user_dir = os.path.join(EMBEDDINGS_DIR, username)
    if not os.path.exists(user_dir):
        return []
    
    embeddings = []
    for f in os.listdir(user_dir):
        emb = np.load(os.path.join(user_dir, f))
        embeddings.append(emb)
    
    return np.array(embeddings)

def augment_image(img):
    """Génère des versions augmentées de l'image (rotation, flip, contraste)."""
    augmentations = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(10),
        img.rotate(-10),
        ImageEnhance.Contrast(img).enhance(1.5),
    ]
    return augmentations

def compute_embeddings(img):
    """
    Calcule les embeddings faciaux à partir d'une image et de ses augmentations.
    Retourne une liste d'embeddings normalisés.
    """
    augmented = augment_image(img)
    embeddings = []

    for aug in augmented:
        face = mtcnn(aug)
        if face is not None:
            with torch.no_grad():
                emb = facenet(face.unsqueeze(0).to(device))
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # Normalisation L2
                embeddings.append(emb.squeeze(0).cpu().numpy())
    
    return embeddings