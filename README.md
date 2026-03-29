# Facial Emotion Recognition - Project

This project is a simple **Facial Emotion Recognition System** built with Python, OpenCV and DeepFace.  
It can:

- preprocess face images,
- detect faces from webcam frames,
- classify emotions,
- visualize emotion results in real time,
- run batch analysis on a dataset,
- save prediction results to a CSV file,
- generate a confusion matrix for accuracy evaluation.

## Members

### Partner A — Meirkhan Amirkhan
Responsible for:
- image preprocessing
- face detection
- webcam mode

### Partner B — Chaizada Arlan
Responsible for:
- emotion classification
- visualization
- batch accuracy analysis

## Features

### 1. Image Preprocessing
The project uses:
- grayscale conversion,
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

to improve image quality before face detection.

### 2. Face Detection
Faces are detected using OpenCV Haar Cascade:
- detects faces in the frame
- returns the biggest face if multiple faces are found
- handles cases when no face is detected

### 3. Emotion Analysis
The project uses DeepFace to analyze emotions such as:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

### 4. Real-Time Webcam Detection
The webcam mode:
- reads live camera input
- detects the face
- predicts the emotion
- draws a bounding box around the face
- shows the dominant emotion
- displays emotion confidence bars
- shows FPS on screen

### 5. Batch Analysis
The system can analyze a folder of categorized images:
- reads images from subfolders,
- predicts emotion for each image,
- compares actual and predicted labels,
- saves results in `results.csv`,
- generates `confusion_matrix.png`.

## What You Need to Download

Before running the program, make sure you have installed:

- **Python 3.10 or 3.11**
- Required Python libraries:
  - opencv-python, pandas, matplotlib, deepface, scikit-learn, tensorflow 

You can install all libraries with:

` pip install opencv-python pandas matplotlib deepface scikit-learn tensorflow ` In terminal


## Work with program
After all this thing, you can run a program with button or:

` python main.py ` In terminal

***

Press `Q` - for quiting and press `S` - for screenshot

## Project Structure

```
project_folder
│
├── main.py
├── data
│   ├── angry
│   │   ├── Image (angry)
│   │   └── ...
│   ├── disgust
│   │   ├── Image (disgust)
│   │   └── ...
│   ├── fear
│   ├── happy
│   └── ...
│
├── results.csv
├── confusion_matrix.png
└── README.md