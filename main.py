import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

''' Partner A: Image preprocessing, face detection, webcam mode
    Meirkhan Amirkhan '''


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(img, scaleFactor=1.3, minNeighbors=5):
# A2
    processed = preprocess_image(img)
    faces = face_cascade.detectMultiScale(
        processed,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )
    if len(faces) == 0:
        return []
    biggest_face = max(faces, key=lambda f: f[2] * f[3])
    return [biggest_face]


''' Partner B: Emotion classification, visualization, batch accuracy analysis
    Chaizada Arlan '''


def analyze_emotions(image_input):
# B1
    try:
        results = DeepFace.analyze(
            img_path=image_input,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            align=True
        )
        return results[0]
    except Exception as e:
        print(f"Error during DeepFace analysis: {e}")
        return None


def draw_results(frame, analysis):
# B2
    if not analysis:
        return frame

    region = analysis['region']
    emotions = analysis['emotion']
    dominant = analysis['dominant_emotion']

    color = (0, 255, 0) if dominant == 'happy' else (0, 0, 255)
    cv2.rectangle(frame, (region['x'], region['y']),
                  (region['x'] + region['w'], region['y'] + region['h']), color, 2)

    cv2.putText(frame, f"{dominant.upper()}", (region['x'], region['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for i, (emo, score) in enumerate(emotions.items()):
        y_offset = region['y'] + (i * 20)
        bar_width = int(score)
        cv2.rectangle(frame, (region['x'] + region['w'] + 10, y_offset),
                      (region['x'] + region['w'] + 110, y_offset + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (region['x'] + region['w'] + 10, y_offset),
                      (region['x'] + region['w'] + 10 + bar_width, y_offset + 15), color, -1)
        cv2.putText(frame, f"{emo}: {score:.1f}%", (region['x'] + region['w'] + 115, y_offset + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def run_batch_analysis(data_folder):
# B3
    all_data = []
    if not os.path.exists(data_folder):
        print(f"Directory {data_folder} not found.")
        return

    for category in os.listdir(data_folder):
        cat_path = os.path.join(data_folder, category)
        if not os.path.isdir(cat_path): continue

        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            res = analyze_emotions(img)
            if res:
                all_data.append({
                    "Filename": img_name,
                    "Actual": category.lower(),
                    "Predicted": res['dominant_emotion'].lower(),
                    "Correct": category.lower() == res['dominant_emotion'].lower()
                })

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("results.csv", index=False)  # Сохранение CSV

        labels = sorted(df['Actual'].unique())
        cm = confusion_matrix(df['Actual'], df['Predicted'], labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Batch Analysis Accuracy: {df['Correct'].mean() * 100:.2f}%")
        plt.savefig("confusion_matrix.png")
        plt.show()


def run_webcam():
# A3
    cap = cv2.VideoCapture(0)
    prev_time = 0
    frame_count = 0
    last_analysis = None
    session_emotions = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        if frame_count % 3 == 0:
            faces = detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                res = analyze_emotions(frame[y:y + h, x:x + w])
                if res:
                    res['region'] = {'x': x, 'y': y, 'w': w, 'h': h}
                    last_analysis = res
                    session_emotions.append(res['dominant_emotion'])
            else:
                last_analysis = None

        if last_analysis:
            frame = draw_results(frame, last_analysis)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit key
        elif key == ord('s'):  # Screenshot key
            cv2.imwrite(f"shot_{frame_count}.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()

    if session_emotions:
        plt.figure(figsize=(8, 5))
        pd.Series(session_emotions).value_counts().plot(kind='bar', color='skyblue')
        plt.title("Session Emotion Distribution")
        plt.ylabel("Frames Detected")
        plt.show()


if __name__ == "__main__":
        run_batch_analysis("data")
        run_webcam()