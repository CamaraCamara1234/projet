# %pip install deepface
from deepface import DeepFace

# Comparer deux images
result = DeepFace.verify(img1_path="./images/photo.png",
                         img2_path="./images/photo3.jpg")
print(f"Les images représentent-elles la même personne ? {result['verified']}")


class RecognationService:
    pass