Face Authentication API

A FastAPI-based face verification service using InsightFace.

Features
Upload two face images
Detect faces
Extract embeddings
Compute similarity score
Verify whether both images belong to the same person
Return face bounding boxes

Tech Stack
Python
FastAPI
InsightFace
OpenCV

installation 
git clone https://github.com/your-username/face-authentication.git cd face-authentication

CODE 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

app = FastAPI(
    title="Face Authentication API",
    description="Face verification using InsightFace",
    version="1.0.0"
)

# Initialize InsightFace model
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0, det_size=(640, 640))


def read_image(file: UploadFile):
    image_bytes = file.file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    return image


def extract_face_data(image):
    """
    Detect faces and extract embeddings.
    Returns:
        faces_data: list of dict
    """

    faces = face_app.get(image)

    if len(faces) == 0:
        return []

    results = []

    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        embedding = face.embedding.tolist()

        results.append({
            "bbox": bbox,
            "embedding": embedding
        })

    return results


def compare_embeddings(emb1, emb2):
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)

    similarity = cosine_similarity(emb1, emb2)[0][0]

    return float(similarity)


@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    Verify whether two images belong to the same person.
    """

    img1 = read_image(image1)
    img2 = read_image(image2)

    faces1 = extract_face_data(img1)
    faces2 = extract_face_data(img2)

    if len(faces1) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in image1"
        )

    if len(faces2) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in image2"
        )

    # Use first detected face
    face1 = faces1[0]
    face2 = faces2[0]

    similarity_score = compare_embeddings(
        face1["embedding"],
        face2["embedding"]
    )

    # Threshold can be tuned
    threshold = 0.5

    result = (
        "same person"
        if similarity_score >= threshold
        else "different person"
    )

    return JSONResponse({
        "verification_result": result,
        "similarity_score": round(similarity_score, 4),
        "threshold_used": threshold,
        "image1_faces": [
            {
                "bounding_box": f["bbox"]
            }
            for f in faces1
        ],
        "image2_faces": [
            {
                "bounding_box": f["bbox"]
            }
            for f in faces2
        ]
    })


@app.get("/")
async def root():
    return {
        "message": "Face Authentication API is running"
    }
