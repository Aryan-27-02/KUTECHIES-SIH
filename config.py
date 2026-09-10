import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STORAGE_DIR = BASE_DIR / "storage"
OUTPUT_DIR = BASE_DIR / "output"
DEMO_DIR = BASE_DIR / "demo_assets"
DATASET_DIR = BASE_DIR / "dataset"

# Ensure directories exist
for p in [MODELS_DIR, STORAGE_DIR, OUTPUT_DIR, DEMO_DIR, DATASET_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# File Paths
AUDITS_CSV = STORAGE_DIR / "audits.csv"
DEFECTS_CSV = STORAGE_DIR / "defects.csv"
EVIDENCE_DIR = OUTPUT_DIR / "evidence_frames"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

KERAS_MODEL_PATH = MODELS_DIR / "condition_classifier.keras"
YOLO_MODEL_NAME = "yolov8n.pt"  # Ultra-fast nano model for real-time edge/laptop audit

# 5 Core Pedestrian Safety Parameters & Rubric Weights
# Reference: Urban School Zone Road Safety Rubric (IRC:67 & IRC:103 Guidelines)
SAFETY_PARAMETERS = {
    "zebra_crossing": {
        "id": "PARAM_1",
        "name": "Zebra Crossing Condition",
        "weight": 0.25,
        "description": "Zebra crossing visibility, paint fading, or missing marking in front of school entrance.",
        "icon": "🦓",
        "classes": ["Intact / Clear", "Worn / Faded (Defect)", "Missing / Absent (Defect)"]
    },
    "footpath_obstruction": {
        "id": "PARAM_2",
        "name": "Footpath Walkability & Encroachment",
        "weight": 0.20,
        "description": "Parked motorcycles, vendors, or debris forcing children to walk on active vehicle carriageway.",
        "icon": "🚶",
        "classes": ["Clear / Usable", "Partially Encroached (Warning)", "Severely Blocked (Defect)"]
    },
    "school_signage": {
        "id": "PARAM_3",
        "name": "School Zone Signage Presence",
        "weight": 0.20,
        "description": "Regulatory and cautionary school warning sign presence and sightline visibility.",
        "icon": "🚸",
        "classes": ["Clearly Visible", "Damaged / Obscured (Warning)", "Missing Sign (Defect)"]
    },
    "obstructing_parking": {
        "id": "PARAM_4",
        "name": "Blind-Spot Obstructing Parking",
        "weight": 0.20,
        "description": "Vehicles parked within the 30m dispersal buffer blocking driver sightlines to exiting children.",
        "icon": "🚗",
        "classes": ["No Obstruction", "Moderate Obstruction (Warning)", "Critical Sightline Blocked (Defect)"]
    },
    "speed_calming": {
        "id": "PARAM_5",
        "name": "Speed Calming & Surface Integrity",
        "weight": 0.15,
        "description": "Raised speed table/breaker condition and critical road surface hazards near school gate.",
        "icon": "🛑",
        "classes": ["Calmed & Good Surface", "Worn Breaker / Minor Hazard (Warning)", "No Speed Calming / Severe Defect (Defect)"]
    }
}

# Scoring Thresholds
RISK_LEVELS = {
    "SAFE": {"min_score": 80, "color": "#059669", "badge": "SAFE / LOW RISK", "desc": "School zone meets basic pedestrian safety standards."},
    "MODERATE": {"min_score": 50, "color": "#D97706", "badge": "MODERATE RISK", "desc": "Identified defects need scheduled municipal maintenance."},
    "HIGH": {"min_score": 0, "color": "#DC2626", "badge": "CRITICAL RISK", "desc": "Immediate civil interventions required (missing markings, blocked walkways)."}
}

# Detection Configuration
YOLO_CONFIDENCE_THRESHOLD = 0.35
FRAME_SAMPLE_INTERVAL = 3  # Process every 3rd frame (roughly 5-10 fps depending on source)
DEFAULT_GPS_LAT = 18.5204   # Sample latitude (Pune/Mumbai or configurable)
DEFAULT_GPS_LON = 73.8567   # Sample longitude

