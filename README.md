# Face Recognition Backend

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Two Ways to Register Your Face

### Option 1: Use the Web App (Recommended)
1. Start the Flask server:
```bash
python app.py
```
2. Open the Angular app in browser
3. Click "Start Camera" and allow camera access
4. Click "Register Face" button
5. Smile and let it capture your face
6. Your face encoding is automatically saved

### Option 2: Train from Existing Images
If you have face photos on your computer:

1. Create the training folder:
```bash
mkdir training_faces
```

2. Add your face images:
   - Place `.jpg`, `.jpeg`, or `.png` files in `backend/training_faces/`
   - You can use 1+ photos of your face from different angles
   - Higher quality = better recognition

3. Run the training script:
```bash
python train_face.py
```

4. Start the Flask server:
```bash
python app.py
```

## File Structure
```
backend/
├── app.py                    # Flask server
├── train_face.py            # Train from existing images
├── requirements.txt         # Python dependencies
├── reference_face.npy       # Stored face encoding (auto-generated)
└── training_faces/          # Place your face images here
    ├── photo1.jpg
    ├── photo2.jpg
    └── photo3.png
```

## API Endpoints

### Health Check
- **GET** `/health` - Check if server is running

### Register Face
- **POST** `/api/register`
- Body: `{ "image": "base64_encoded_image" }`
- Registers a face from the web app webcam

### Verify Face
- **POST** `/api/verify`
- Body: `{ "image": "base64_encoded_image" }`
- Verifies if the provided face matches the registered reference face
- Returns: `{ "verified": boolean, "confidence": number }`

## Notes

- The reference face encoding is stored in `reference_face.npy` (binary file, not images)
- Face detection tolerance can be adjusted in `app.py` verify endpoint (default: 0.6)
- Lower tolerance = stricter matching
- Multiple training images are averaged for better accuracy
