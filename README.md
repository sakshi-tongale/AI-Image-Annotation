# AI Image Annotation

## Overview
AI Image Annotation is a Python-based application that uses YOLO11 for automatic object detection, instance-level labeling, and object descriptions. It also supports manual bounding box annotation and exporting annotations in JSON and YOLO formats.

## Features
- Automatic object detection using YOLO11.
- Instance-level labeling (Person 1, Person 2, Chair 1, etc.).
- Individual object descriptions.
- Automatic scene description.
- Manual bounding box creation and editing.
- Add, update, and delete object annotations.
- Final review of all detected objects.
- Export annotations in JSON format.
- Export annotations in YOLO format.
- Download annotated images.

## Technologies Used
- Python
- Streamlit
- YOLO11
- Ultralytics
- Pillow
- Streamlit Drawable Canvas

## Project Structure
```text
AI-Image-Annotation/
│
├── app.py
├── yolo11n.pt
├── requirements.txt
├── README.md
├── .gitignore
│
├── sample_images/
├── screenshots/
└── output/
```

## Installation

### 1. Clone the Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open Project Folder
```bash
cd AI-Image-Annotation
```

### 3. Create Virtual Environment
```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows:
```bash
venv\Scripts\activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Run the Application
```bash
streamlit run app.py
```

## How It Works
1. Upload an image.
2. Run automatic object detection.
3. View detected objects and their bounding boxes.
4. Review individual object labels and descriptions.
5. Add or edit objects manually if required.
6. Review final annotations.
7. Export annotations in JSON or YOLO format.

## Output Formats
- JSON – Structured object annotation data.
- YOLO – Object detection annotation format.
- Annotated Image – Image with bounding boxes and labels.

## Applications
- Computer Vision
- Object Detection
- Image Annotation
- AI and Machine Learning
- Dataset Preparation


