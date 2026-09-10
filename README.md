# SafeStreet: A Smartphone Vision System for Auditing Pedestrian Safety in Urban School Zones

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3-red.svg)](https://keras.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](SafeStreet_Model_Training.ipynb)

SafeStreet is an automated, AI-driven road safety auditing system designed for urban school zones. It processes ordinary smartphone video—captured from a two-wheeler mount or by an auditor walking past a school—to automatically detect, classify, geotag, and score pedestrian safety defects against an established civil engineering rubric.

The system aggregates audits into a **ranked municipal priority list**, giving city corporations an objective, evidence-backed tool to decide which school zones need maintenance and safety funds first.

---

## 🏛️ Conceptual Architecture

SafeStreet strictly decouples perception, policy, and decision-making into three modular tiers:

$$\textbf{Computer Vision Detects} \longrightarrow \textbf{Rule Engine Interprets} \longrightarrow \textbf{Scoring Engine Evaluates}$$

```
                ┌──────────────────────────────────┐
                │ Smartphone Video (.mp4/.mov/avi) │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │ Video to Frames: OpenCV Pipeline │
                └─────────────────┬────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│     Object Detection: YOLO    │         │ Condition Classifier: Keras/TF│
│ (Pedestrians, Two-Wheelers,   │         │ (Zebra Crossing Degradation,  │
│  Parked Cars, School Signs)   │         │  Sign Visibility & Occlusion) │
└───────────────┬───────────────┘         └───────────────┬───────────────┘
                │                                         │
                └─────────────────┬───────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │    Rule Engine (IRC-67 & 103)    │
                │ Maps CV to Safe / Warning / Defect│
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │  Scoring Engine (0-100 Scale)    │
                │  - School Safety Index (SSI)     │
                │  - Municipal Urgency Priority    │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │ Streamlit Dashboard & Evidence   │
                │ - Live Annotated HUD Video       │
                │ - Geotagged Defect Gallery       │
                │ - Municipal Ranking Table        │
                │ - Exportable Civil Work Orders   │
                └──────────────────────────────────┘
```

---

## 🚦 5-Parameter Safety Rubric

SafeStreet measures school zone safety across 5 critical parameters based on Indian Road Congress (IRC:67 & IRC:103) and global school-zone safety guidelines:

| # | Parameter | Detection / Classification Source | Defect Threshold | Weight |
|---|---|---|---|---|
| **1** | **Zebra Crossing Condition** | Keras/TF Classifier + OpenCV Road ROI | Paint wear $>50\%$ or missing designated crossing | **25%** |
| **2** | **Footpath Walkability** | YOLO on Sidewalk Spatial Zones | Two-wheelers or vehicles parked on pedestrian path | **20%** |
| **3** | **School Zone Signage** | YOLO (Sign Detector) + Keras (Clarity) | Absent or obscured/damaged school warning sign | **20%** |
| **4** | **Blind-Spot Parking** | YOLO (Stationary Vehicles in 30m buffer) | Cars/vans parked blocking line-of-sight to children | **20%** |
| **5** | **Speed Calming & Surface** | Keras/TF Surface / Edge Density | Missing speed calming device or severe road defects | **15%** |

### Risk Level Categorization:
- 🟢 **SAFE / LOW RISK (80 - 100)**: Compliant with school zone safety requirements.
- 🟡 **MODERATE RISK (50 - 79)**: Minor issues; needs scheduled municipal maintenance.
- 🔴 **HIGH / CRITICAL RISK (0 - 49)**: Severe hazards present (missing crossing, blocked footpath); requires immediate civic intervention.

---

## 📂 Project Structure

```
e:\SIH\Project\
├── app.py                      # Interactive Streamlit Municipal Dashboard
├── config.py                   # Central configuration, paths, thresholds, rubric weights
├── requirements.txt            # Python dependencies
├── SafeStreet_Model_Training.ipynb # Google Colab Notebook for Keras & YOLO training
├── README.md                   # Complete documentation
│
├── core/
│   ├── video_processor.py      # OpenCV video ingestion, frame extraction, HUD rendering
│   ├── object_detector.py      # YOLO detector (pedestrians, vehicles, two-wheelers, signs)
│   ├── condition_classifier.py # Keras + TensorFlow road & crossing degradation classifier
│   ├── rule_engine.py          # Interprets CV data into SAFE / WARNING / CRITICAL DEFECT
│   └── scoring_engine.py       # Computes School Safety Index (SSI) & Priority Score
│
├── storage/
│   ├── audit_store.py          # CSV storage manager for audits and defect logs
│   ├── audits.csv              # Audit records database
│   └── defects.csv             # Defect records with geotags and timestamps
│
├── train/
│   ├── generate_synthetic_data.py # Synthesizes labeled training samples for Keras
│   └── train_keras.py          # Standalone Keras/TF training script
│
├── models/
│   └── condition_classifier.keras # Exported neural network model
│
├── demo_assets/
│   ├── generate_sample_video.py# Script to generate realistic test school zone video
│   └── sample_school_zone.mp4  # Pre-rendered test video
│
└── output/
    └── evidence_frames/        # Geotagged high-resolution defect snapshots
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Generate Demo Assets & Train Model
```bash
# Generate synthetic dataset and train the Keras condition classifier
python train/train_keras.py

# Generate realistic test video for instant hackathon demonstration
python demo_assets/generate_sample_video.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## ☁️ Google Colab Training

To train on real-world imagery or fine-tune YOLO on localized datasets:
1. Open `SafeStreet_Model_Training.ipynb` in Google Colab.
2. Follow the step-by-step cells for data augmentation, MobileNetV2 transfer learning, and YOLO fine-tuning.
3. Download `condition_classifier.keras` and place it in the `models/` directory.

---

## 🏆 Key Strengths for Hackathon Judges

1. **Decoupled 3-Tier Architecture**: Perception is separated from policy and economics. Changes in municipal guidelines do not break or require retraining the vision models.
2. **Actionable Civic Output**: Instead of just drawing bounding boxes, SafeStreet outputs a **Municipal Intervention Priority Ranking**, answering the exact question city engineers care about: *"Which school zone should we allocate road repair funds to first?"*
3. **Dual AI Model Synergy**: Blends YOLO (fast spatial object detection) with Keras/TensorFlow (nuanced condition/texture analysis of paint wear and signs).
4. **Lightweight & Edge-Ready**: Runs on standard laptops or edge devices using CPU/GPU with fast frame sampling.

