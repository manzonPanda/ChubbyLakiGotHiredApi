from deepface import DeepFace
import os
import numpy as np
from PIL import Image

TRAINING_DIR = 'training_faces/Dianne'
# TRAINING_DIR = 'training_faces/Jake'
REFERENCE_FACE_PATH = 'reference_face.npy'

def train_from_images():
    """Train face recognition model from images in training_faces directory"""
    
    if not os.path.exists(TRAINING_DIR):
        print(f"Creating {TRAINING_DIR} directory...")
        os.makedirs(TRAINING_DIR)
        print(f"Please add your face images to the {TRAINING_DIR}/ folder")
        return
    
    images = [f for f in os.listdir(TRAINING_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"No images found in {TRAINING_DIR}/")
        print("Please add .jpg, .jpeg, or .png files to train the model")
        return
    
    print(f"Found {len(images)} image(s) in {TRAINING_DIR}/")
    
    face_encodings = []

    for image_file in images:
        image_path = os.path.join(TRAINING_DIR, image_file)
        print(f"Processing {image_file}...")

        try:
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name='ArcFace',
                detector_backend='opencv'
            )

            if len(embeddings) == 0:
                print(f"  ⚠ No face detected in {image_file}")
            else:
                print(f"  ✓ Face detected in {image_file}")

            if isinstance(embeddings[0], dict):
                embedding = embeddings[0]["embedding"]
            else:
                embedding = embeddings
            face_encodings.append(np.array(embedding, dtype=np.float32))
            print("Embedding length:",len(embedding))
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
    
    if face_encodings:
        # Average multiple encodings if multiple images provided
        reference_encodings = np.array([
            emb / np.linalg.norm(emb)
            for emb in face_encodings
        ])
        np.save(
            REFERENCE_FACE_PATH,
            reference_encodings
        )

        print(f"Saved {len(reference_encodings)} embeddings")
        print(f"\n✓ Face model trained and saved to {REFERENCE_FACE_PATH}")
        print(f"  Used {len(face_encodings)} face encoding(s)")
    else:
        print("\n✗ No valid faces found to train")

if __name__ == '__main__':
    train_from_images()
