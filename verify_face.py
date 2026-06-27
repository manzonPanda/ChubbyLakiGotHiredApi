from deepface import DeepFace
import numpy as np

REFERENCE_FACE_PATH = "reference_face.npy"
TEST_IMAGE = "test.jpg"  # Change this to your test image

try:
    # Load trained face embedding
    reference_embedding = np.load(REFERENCE_FACE_PATH)

    # Generate embedding for test image
    captured_embedding = DeepFace.represent(
        img_path=TEST_IMAGE,
        model_name="Facenet512",
        enforce_detection=True
    )

    # Handle different DeepFace return formats
    if isinstance(captured_embedding[0], dict):
        captured_embedding = np.array(captured_embedding[0]["embedding"])
    else:
        captured_embedding = np.array(captured_embedding)

    # Calculate cosine similarity
    similarity = np.dot(
        reference_embedding,
        captured_embedding
    ) / (
        np.linalg.norm(reference_embedding)
        * np.linalg.norm(captured_embedding)
    )

    similarity_percent = similarity * 100

    # Determine quality label
    if similarity_percent >= 90:
        quality = "Excellent Match"
    elif similarity_percent >= 80:
        quality = "Strong Match"
    elif similarity_percent >= 70:
        quality = "Good Match"
    elif similarity_percent >= 60:
        quality = "Weak Match"
    else:
        quality = "No Match"

    # DeepFace official verification
    # Use the original training image here
    result = DeepFace.verify(
        img1_path="training_faces/circle_pic.jpg",
        img2_path=TEST_IMAGE,
        model_name="Facenet512",
        detector_backend="opencv"
    )

    print("\n====================================")
    print("        FACE VERIFICATION")
    print("====================================")
    print(f"Similarity: {similarity_percent:.2f}%")
    print(f"Match Quality: {quality}")
    print(f"Verified: {'YES' if result['verified'] else 'NO'}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Threshold: {result['threshold']}")
    print("====================================")

except Exception as e:
    print(f"Error: {e}")