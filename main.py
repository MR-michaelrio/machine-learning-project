from fastapi import FastAPI, File, UploadFile
from PIL import Image
import face_recognition
import numpy as np
import os
from typing import List
import operator

app = FastAPI()

# machine learning
@app.post("/train_dataset")
async def train_dataset():
    if not os.path.exists("encodings"):
        os.makedirs("encodings")

    dataset_folder = "Humans"
    for file_name in os.listdir(dataset_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
            nim = os.path.splitext(file_name)[0]
            encoding_file = f"encodings/{nim}.txt"
            if not os.path.exists(encoding_file):
                file_path = os.path.join(dataset_folder, file_name)
                image = face_recognition.load_image_file(file_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if len(face_encodings) > 0:
                    with open(encoding_file, "w") as f:
                        for encoding in face_encodings:
                            f.write(" ".join(str(e) for e in encoding) + "\n")

    return {"status": "success"}

@app.post("/face_recognitionml")
async def recognize_faces(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Load the image file and find face locations and encodings
    image = face_recognition.load_image_file(file_location)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Load saved encodings
    encodings = []
    for filename in os.listdir("encodings"):
        if filename.endswith(".txt"):
            with open(os.path.join("encodings", filename), "r") as f:
                name = os.path.splitext(filename)[0]
                lines = f.readlines()
                encoding = [float(x) for x in lines[0].strip().split()]
                encodings.append((name, encoding))

    # Find matches and their accuracies
    matches = []
    accuracies = []
    for encoding in face_encodings:
        best_match = None
        best_accuracy = 0
        for name, saved_encoding in encodings:
            distance = face_recognition.face_distance([saved_encoding], encoding)
            accuracy = (1 - distance[0]) * 100
            if accuracy > best_accuracy:
                best_match = name
                best_accuracy = accuracy

        if best_match is not None:
            matches.append(best_match)
            accuracies.append(best_accuracy)

    os.remove(file_location)

    if len(matches) > 0:
        # Find the highest accuracy match
        highest_accuracy_index, _ = max(enumerate(accuracies), key=operator.itemgetter(1))
        highest_accuracy_match = matches[highest_accuracy_index]

        return {"match": highest_accuracy_match, "accuracy": accuracies[highest_accuracy_index], "face": "Recognize"}
    else:
        return {"face": "NoRecognize"}