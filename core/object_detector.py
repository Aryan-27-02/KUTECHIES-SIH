import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os

class ObjectDetector:
    """
    YOLO-based detector for pedestrians (children/adults), vehicles (cars, motorcycles, buses),
    and road infrastructure signs in school zones.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35):
        self.conf_thresh = conf_thresh
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initializes Ultralytics YOLO with fallback if weights are downloading."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"[SafeStreet] YOLO model loaded: {self.model_name}")
        except Exception as e:
            print(f"[SafeStreet] Warning: YOLO loading deferred or failed ({e}). Fallback detector active.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executes detection on frame and maps spatial zones (footpath vs carriageway).
        Returns a list of detected objects with class, confidence, bounding box, and zone.
        """
        h, w = frame.shape[:2]
        detections = []

        if self.model is not None:
            try:
                results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = results.names.get(cls_id, f"obj_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    
                    zone = self._determine_spatial_zone(x1, y1, x2, y2, w, h)
                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "zone": zone
                    })
                return detections
            except Exception as e:
                print(f"[SafeStreet] YOLO detection runtime error ({e}), using heuristic.")

        # Heuristic / Simulation fallback if YOLO is initializing or unavailable
        return self._heuristic_fallback_detect(frame)

    def _determine_spatial_zone(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
        """
        Determines whether the object is in:
        - 'footpath_left' (left 22% of frame)
        - 'footpath_right' (right 22% of frame)
        - 'carriageway' (central roadway)
        - 'crossing_approach' (lower central region)
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if cy > h * 0.55 and (w * 0.25 <= cx <= w * 0.75):
            return "crossing_approach"
        elif cx < w * 0.22:
            return "footpath_left"
        elif cx > w * 0.78:
            return "footpath_right"
        else:
            return "carriageway"

    def _heuristic_fallback_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Heuristic detector using color and contour thresholds for demo continuity
        when YOLO weights or packages are in cold-start.
        """
        h, w = frame.shape[:2]
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect bright white stripes (Zebra crossing / road markings) in lower half
        road_roi = gray[int(h*0.5):, :]
        _, thresh = cv2.threshold(road_roi, 190, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stripes = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 40000:
                stripes += 1
                
        if stripes >= 3:
            detections.append({
                "class": "zebra_crossing",
                "confidence": 0.88,
                "bbox": [int(w*0.2), int(h*0.6), int(w*0.8), int(h*0.92)],
                "zone": "crossing_approach"
            })

        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws bounding boxes and labels on frame with color distinction.
        """
        annotated = frame.copy()
        
        # Distinct palette
        color_map = {
            "person": (60, 220, 100),       # Green for pedestrians
            "motorcycle": (255, 120, 50),   # Cyan/Orange for two-wheelers
            "car": (240, 80, 80),           # Blue for cars
            "bus": (200, 50, 230),          # Purple for buses
            "traffic sign": (30, 210, 245),  # Yellow for signs
            "stop sign": (30, 210, 245),
            "zebra_crossing": (240, 240, 240)
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            conf = det["confidence"]
            zone = det.get("zone", "")
            
            color = color_map.get(cls, (180, 180, 180))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls} {int(conf*100)}%"
            if zone in ["footpath_left", "footpath_right"] and cls in ["motorcycle", "car", "bicycle"]:
                label += " [ON FOOTPATH!]"
                color = (40, 40, 235)  # Highlight in Red
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Text background badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

