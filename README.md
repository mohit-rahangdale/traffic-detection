# 🚦 Traffic Vehicle Detection System

This project implements a computer vision system that **automatically detects, classifies, and counts vehicles** in traffic images. It uses **YOLOv8** (pre-trained) to detect **cars**, **motorcycles**, and **trucks** with high accuracy, draws bounding boxes with confidence scores, and saves annotated output images.

---

## 📌 Features

- ✅ Detects **cars, trucks, motorcycles**
- ✅ Draws **bounding boxes** with **labels** and **confidence scores**
- ✅ Counts total vehicles by category
- ✅ Saves **processed images** with annotations
- ✅ Uses pre-trained **YOLOv8** for robust results
- ✅ Simple, clean Python code (OpenCV, NumPy, Ultralytics)

---

## ⚙️ Technology Stack

- **Language:** Python 3.8+
- **Libraries:** OpenCV, NumPy, Matplotlib, Torch, Ultralytics
- **ML Model:** YOLOv8l (`ultralytics` library)
- **Dataset:** COCO pre-trained (vehicle classes)

---

## 📂 Project Structure

```text
traffic-detection-assignment/
├── README.md
├── requirements.txt
├── main.py
├── detector.py
├── utils.py
├── data/
│ └── test_images/ #test images here
├── output/
│ └── processed_images/ # Annotated images will be saved here
└── docs/
├── technical_report.pdf
└── presentation_slides.pdf

## ⚡ Setup & Installation (with uv)

```bash
uv pip install -r requirements.txt
python main.py

