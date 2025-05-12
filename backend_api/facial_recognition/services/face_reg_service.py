# Importations complètes
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from django.conf import settings
import os
import json
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

# ----------------------------------------------------------
# CONFIGURATION AVANCÉE DE DÉTECTION DE VISAGE
# ----------------------------------------------------------

class FaceDetector:
    def __init__(self, method='mtcnn'):
        self.method = method
        if method == 'haar':
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            self.detector = MTCNN()

    def detect(self, image):
        if self.method == 'haar':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))
            return [(x, y, w, h) for (x, y, w, h) in faces]
        else:
            results = self.detector.detect_faces(image)
            return [result['box'] for result in results]

# ----------------------------------------------------------
# PRÉTRAITEMENT AVANCÉ DES IMAGES
# ----------------------------------------------------------

class FacePreprocessor:
    def __init__(self):
        self.detector = FaceDetector(method='mtcnn')
        self.target_size = (100, 100)
    
    def process(self, img_path):
        try:
            # Chargement de l'image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Image non trouvée ou corrompue")
            
            # Conversion RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Détection du visage
            faces = self.detector.detect(img)
            if not faces:
                raise ValueError("Aucun visage détecté")
            
            # Sélection du plus grand visage
            main_face = max(faces, key=lambda f: f[2]*f[3])
            x, y, w, h = main_face
            
            # Recadrage et redimensionnement
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, self.target_size)
            
            # Contrôle de qualité
            self._check_quality(face)
            
            # Normalisation
            face = face.astype('float32') / 255.0
            
            return face
        except Exception as e:
            print(f"Erreur de prétraitement {img_path}: {str(e)}")
            raise

    def _check_quality(self, face):
        # Vérification de la netteté
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            raise ValueError("Image trop floue (variance Laplacienne < 50)")

# ----------------------------------------------------------
# ARCHITECTURE SIAMOISE AVANCÉE
# ----------------------------------------------------------

class SiameseNetwork:
    def __init__(self):
        self.input_shape = (100, 100, 3)
        self.model = self.build_network()
    
    def build_network(self):
        # Réseau de base
        base_cnn = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Couches personnalisées
        inputs = layers.Input(self.input_shape)
        x = base_cnn(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        
        embedding_model = models.Model(inputs, x)
        
        # Architecture siamoise
        input_1 = layers.Input(self.input_shape)
        input_2 = layers.Input(self.input_shape)
        
        embedding_1 = embedding_model(input_1)
        embedding_2 = embedding_model(input_2)
        
        distance = layers.Lambda(
            lambda embeddings: tf.math.abs(embeddings[0] - embeddings[1])
        )([embedding_1, embedding_2])
        
        outputs = layers.Dense(1, activation='sigmoid')(distance)
        
        siamese_model = models.Model([input_1, input_2], outputs)
        siamese_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=[metrics.BinaryAccuracy(name='acc')]
        )
        
        return siamese_model

# ----------------------------------------------------------
# GESTION DES DONNÉES ET ENTRAÎNEMENT
# ----------------------------------------------------------

class FaceDataManager:
    def __init__(self):
        self.preprocessor = FacePreprocessor()
        self.data_dir = os.path.join(settings.MEDIA_ROOT, "data")
    
    def load_user_data(self, user_id):
        user_dir = os.path.join(self.data_dir, str(user_id))
        faces = []
        
        if os.path.exists(user_dir):
            for img_file in os.listdir(user_dir):
                if img_file.startswith('processed_'):
                    img_path = os.path.join(user_dir, img_file)
                    try:
                        face = self.preprocessor.process(img_path)
                        faces.append(face)
                    except Exception as e:
                        print(f"Erreur chargement {img_path}: {str(e)}")
        return np.array(faces)
    
    def generate_pairs(self, embeddings, n_neg=5):
        pairs = []
        labels = []
        
        # Paires positives
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                pairs.append([embeddings[i], embeddings[j]])
                labels.append(1)
        
        # Paires négatives
        if len(embeddings) > 1:
            for i in range(len(embeddings)):
                others = [j for j in range(len(embeddings)) if j != i]
                for _ in range(n_neg):
                    j = np.random.choice(others)
                    pairs.append([embeddings[i], embeddings[j]])
                    labels.append(0)
        
        return np.array(pairs), np.array(labels)
    
    def incremental_train(self, user_id):
        model_path = os.path.join(settings.BASE_DIR, 'facial_recognition/face_models/face_model.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        user_faces = self.load_user_data(user_id)
        if len(user_faces) < 2:
            return False
            
        siamese_net = SiameseNetwork()
        if os.path.exists(model_path):
            siamese_net.model.load_weights(model_path)
        
        pairs, labels = self.generate_pairs(user_faces)
        
        siamese_net.model.fit(
            [pairs[:,0], pairs[:,1]],
            labels,
            batch_size=8,
            epochs=10,
            verbose=1
        )
        
        siamese_net.model.save_weights(model_path)
        return True
    
    def get_user_image_count(self, user_id):
        user_dir = os.path.join(self.data_dir, str(user_id))
        if not os.path.exists(user_dir):
            return 0
        return len([f for f in os.listdir(user_dir) if f.startswith('processed_')])
    
    def get_total_image_count(self):
        count = 0
        for user_dir in os.listdir(self.data_dir):
            count += self.get_user_image_count(user_dir)
        return count
    
    def should_trigger_global_training(self):
        return self.get_total_image_count() >= 10
# ----------------------------------------------------------
# UTILITAIRES D'ENTRAÎNEMENT
# ----------------------------------------------------------

def full_training():
    data_manager = FaceDataManager()
    model_path = os.path.join(settings.BASE_DIR, 'facial_recognition/face_models/face_model.h5')
    
    try:
        siamese_net = SiameseNetwork()
        if os.path.exists(model_path):
            siamese_net.model.load_weights(model_path)
        
        # Collecter toutes les images valides
        all_faces = []
        for user_dir in os.listdir(data_manager.data_dir):
            user_id = os.path.basename(user_dir)
            faces = data_manager.load_user_data(user_id)
            if len(faces) > 0:
                all_faces.extend(faces)
        
        if len(all_faces) < 2:
            return False, "Pas assez de données (minimum 2 images requises)"
        
        # Générer des paires équilibrées
        pairs, labels = data_manager.generate_pairs(np.array(all_faces))
        
        # Entraînement avec callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_acc')
        siamese_net.model.fit(
            [pairs[:,0], pairs[:,1]],
            labels,
            batch_size=32,
            epochs=20,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Sauvegarde
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        siamese_net.model.save_weights(model_path)
        
        return True, f"Entraînement réussi avec {len(all_faces)} images"
    
    except Exception as e:
        return False, f"Erreur d'entraînement: {str(e)}"
    

    # ----------------------------------------------------------
# VERIFICATION FACIALE AMELIOREE
# ----------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
import datetime

class FaceVerifier:
    def __init__(self):
        self.preprocessor = FacePreprocessor()
        self.model_path = os.path.join(settings.BASE_DIR, 'facial_recognition/face_models/face_model.h5')
        self.embedding_model = self._load_embedding_model()
        
    def _load_embedding_model(self):
        """Charge uniquement la partie embedding du modèle siamois avec vérification"""
        base_model = SiameseNetwork().model.layers[2]  # Couche embedding
        
        if os.path.exists(self.model_path):
            try:
                # Chargement sélectif des poids
                base_model.load_weights(self.model_path, by_name=True)
                # Test de validation
                test_input = np.random.rand(1, 100, 100, 3)
                embedding = base_model.predict(test_input)
                if embedding.shape != (1, 256):  # Vérifie la dimension attendue
                    raise ValueError("Dimension incorrecte des embeddings")
            except Exception as e:
                print(f"Erreur de chargement du modèle: {str(e)}")
                # Reconstruit un modèle vierge si le chargement échoue
                base_model = SiameseNetwork().model.layers[2]
        
        return base_model
    
    def verify_face(self, input_img_path, username):
        try:
            # 1. Validation des entrées
            if not os.path.exists(input_img_path):
                raise ValueError("Fichier image introuvable")
            
            # 2. Chargement des visages de référence
            data_manager = FaceDataManager()
            reference_faces = data_manager.load_user_data(username)
            
            if len(reference_faces) == 0:
                return {
                    'status': 'error',
                    'message': f'Aucune image de référence pour {username}',
                    'verified': False,
                    'confidence': 0.0
                }

            # 3. Prétraitement avec vérification
            input_face = self.preprocessor.process(input_img_path)
            input_embedding = self.embedding_model.predict(np.expand_dims(input_face, axis=0))
            
            # 4. Calcul des similarités avec normalisation
            similarities = []
            for ref_face in reference_faces:
                ref_embedding = self.embedding_model.predict(np.expand_dims(ref_face, axis=0))
                
                # Vérification des embeddings avant calcul
                if np.allclose(input_embedding, 0) or np.allclose(ref_embedding, 0):
                    raise ValueError("Embedding nul détecté")
                
                similarity = cosine_similarity(input_embedding, ref_embedding)[0][0]
                similarities.append(similarity)
            
            # 5. Analyse des résultats
            confidence = np.max(similarities)
            threshold = self._dynamic_threshold(len(reference_faces))
            
            # 6. Journalisation pour débogage
            self._log_verification(input_img_path, username, confidence, threshold)
            
            return {
                'status': 'success',
                'verified': bool(confidence > threshold),
                'confidence': float(confidence),
                'threshold': float(threshold),
                'comparisons': int(len(reference_faces)),
                'best_match': os.path.basename(input_img_path)
            }
            
        except Exception as e:
            self._log_error(input_img_path, str(e))
            return {
                'status': 'error',
                'message': str(e),
                'verified': False,
                'confidence': 0.0
            }

    def _dynamic_threshold(self, num_refs):
        """Seuil adaptatif basé sur la quantité de données"""
        base_threshold = 0.75  # Plus strict pour éviter faux positifs
        return max(0.6, base_threshold - (0.02 * min(5, num_refs - 1)))
    
    def _log_verification(self, img_path, username, confidence, threshold):
        """Journalisation détaillée pour analyse"""
        log_dir = os.path.join(settings.MEDIA_ROOT, 'verification_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'image': os.path.basename(img_path),
            'confidence': confidence,
            'threshold': threshold,
            'decision': 'ACCEPTED' if confidence > threshold else 'REJECTED'
        }
        
        log_file = os.path.join(log_dir, f'{datetime.now().date()}.json')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_error(self, img_path, error_msg):
        """Journalisation des erreurs"""
        error_dir = os.path.join(settings.MEDIA_ROOT, 'verification_errors')
        os.makedirs(error_dir, exist_ok=True)
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'image': os.path.basename(img_path) if img_path else 'unknown',
            'error': error_msg
        }
        
        error_file = os.path.join(error_dir, f'errors_{datetime.now().date()}.json')
        with open(error_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
