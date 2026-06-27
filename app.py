from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import tempfile
# from scipy.spatial.distance import cosine
import threading
import json

app = Flask(__name__)
CORS(app)

# Store reference face encoding (train with admin face)
# reference_face_encoding = None

REFERENCE_FACE_PATH = 'reference_face.npy'
reference_face_embeddings = None

def load_reference_face():
    global reference_face_embeddings
    if os.path.exists(REFERENCE_FACE_PATH):
        reference_face_embeddings = np.load(
            REFERENCE_FACE_PATH,
            allow_pickle=True
        )
        print("Loaded embeddings:",reference_face_embeddings.shape)
        return True
    return False

print("Loading ArcFace model...")
model = DeepFace.build_model("ArcFace")
print("Model loaded")

load_reference_face()

def save_reference_face(face_encoding):
    """Save reference face encoding for future comparisons"""
    np.save(REFERENCE_FACE_PATH, face_encoding)

def base64_to_image_file(image_data):
    """Convert base64 image to temporary file path"""
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(temp_file.name, 'JPEG')
        temp_file.close()

        return temp_file.name
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def cosine_distance(a, b):
    similarity = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return 1 - similarity

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/api/verify', methods=['POST'])
def verify_face():
    """Verify if the captured face matches the registered reference face"""
    temp_file = None
    try:
        if reference_face_embeddings is None:
            return jsonify({'error': 'No reference face registered'}), 400

        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        temp_file = base64_to_image_file(image_data)
        if temp_file is None:
            return jsonify({'error': 'Failed to process image'}), 400

        try:
            # Extract face embedding from capture
            result = DeepFace.represent(
                img_path=temp_file,
                model_name='ArcFace'
            )

            if isinstance(result[0], dict):
                embedding = np.array(result[0]["embedding"])
            else:
                embedding = np.array(result)
            # embedding = np.array(embedding)

            # Calculate cosine distance between embeddings
            # DeepFace uses cosine distance; threshold of 0.4 is commonly used for Facenet512
            distances = []

            for ref_embedding in reference_face_embeddings:
                dist = cosine_distance(
                    ref_embedding,
                    embedding
                )
                distances.append(dist)

            distances.sort()

            best_distance = min(distances)
            
            threshold = 0.25

            is_verified = best_distance < threshold

            similarity = 1 - best_distance
            match_percentage = similarity * 100

            return jsonify({
                'verified': is_verified,
                'similarity': round(match_percentage, 2),
                'distance': float(best_distance),
                'message': 'Face verified successfully' if is_verified else 'Face does not match'
            })
        except ValueError:
            return jsonify({'error': 'No face detected in image'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

@app.route('/api/quick-detect', methods=['POST'])
def quick_face_detect():
    """Quick face detection and match confidence for real-time feedback"""
    temp_file = None
    try:
        if reference_face_embeddings is None:
            return jsonify({'face_detected': False, 'confidence': 0,'debug': str(reference_face_embeddings is not None)}), 200

        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'face_detected': False, 'confidence': 0}), 400

        temp_file = base64_to_image_file(image_data)
        if temp_file is None:
            return jsonify({'face_detected': False, 'confidence': 0}), 400

        try:
            # Quick face detection with embedding
            results = DeepFace.represent(
                img_path=temp_file,
                model_name='ArcFace',
                detector_backend='retinaface'
            )

            if not results or len(results) == 0:
                return jsonify({
                    'face_detected': False,
                    'confidence': 0
                })

            # Handle both DeepFace formats
            if isinstance(results[0], dict):
                embedding = np.array(results[0]["embedding"])
            else:
                embedding = np.array(results)
            distances = [
                cosine_distance(ref, embedding)
                for ref in reference_face_embeddings
            ]

            distances.sort()

            best_distance = min(distances)

            similarity = 1 - best_distance

            match_percentage = round(similarity * 100, 1)

            confidence = similarity

            # Determine match quality
            if match_percentage >= 90:
                quality = "Excellent Match"
            elif match_percentage >= 80:
                quality = "Strong Match"
            elif match_percentage >= 70:
                quality = "Good Match"
            elif match_percentage >= 60:
                quality = "Weak Match"
            else:
                quality = "No Match"

            return jsonify({
                'face_detected': True,
                'confidence': confidence,
                'distance': float(best_distance),
                'quality': quality,
                'match_percentage': round(confidence * 100, 1)
            })
        except (ValueError, Exception) as e:
            return jsonify({'face_detected': False, 'confidence': 0, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'face_detected': False, 'error': str(e)}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
