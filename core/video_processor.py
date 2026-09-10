import cv2
import os
import time
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional
import numpy as np

class VideoProcessor:
    """
    Handles smartphone video intake, frame extraction, metadata extraction,
    and simulated/extracted GPS telemetry for SafeStreet audits.
    """
    def __init__(self, video_path: str, sample_interval: int = 3):
        self.video_path = video_path
        self.sample_interval = max(1, sample_interval)
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source at {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "filename": Path(self.video_path).name,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "resolution": f"{self.width}x{self.height}",
            "duration_sec": round(self.duration_sec, 2),
            "sampled_frames_expected": self.total_frames // self.sample_interval
        }

    def frame_generator(
        self, 
        base_lat: float = 18.5204, 
        base_lon: float = 73.8567
    ) -> Generator[Tuple[int, float, float, float, np.ndarray], None, None]:
        """
        Yields sampled frames with synchronized timestamp and simulated/interpolated GPS coordinates:
        (frame_index, timestamp_sec, lat, lon, frame_bgr)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        # Simulate gentle movement along a street (e.g. walking / two-wheeler audit at ~10-15 km/h)
        lat_step = 0.00002  # ~2.2 meters per step
        lon_step = 0.000015

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
                
            if frame_idx % self.sample_interval == 0:
                timestamp = frame_idx / self.fps
                current_lat = base_lat + (frame_idx / 100.0) * lat_step
                current_lon = base_lon + (frame_idx / 100.0) * lon_step
                yield frame_idx, timestamp, current_lat, current_lon, frame
                
            frame_idx += 1

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    @staticmethod
    def draw_hud(frame: np.ndarray, title: str, status_text: str, score: float, risk_level: str) -> np.ndarray:
        """
        Draws an informative HUD overlay on the video frame showing parameters,
        School Safety Index, and Risk level badge.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Header banner (dark translucent)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 24, 33), -1)
        # Footer banner
        cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        
        # Color coding for risk
        color_map = {
            "SAFE": (70, 190, 80),      # Green
            "MODERATE": (30, 175, 245), # Amber / Yellow
            "HIGH": (40, 50, 235)       # Red
        }
        badge_color = color_map.get(risk_level, (200, 200, 200))
        
        # Top HUD Text
        cv2.putText(annotated, "SafeStreet AI | School Zone Safety Audit", (20, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Target: {title}", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 190, 205), 1, cv2.LINE_AA)
        
        # Risk Badge on Top Right
        badge_text = f"SSI: {score:.1f}/100 [{risk_level}]"
        cv2.rectangle(annotated, (w - 260, 12), (w - 20, 52), badge_color, -1)
        cv2.putText(annotated, badge_text, (w - 245, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Bottom HUD Text
        cv2.putText(annotated, f"Status: {status_text}", (20, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
                    
        return annotated

    @staticmethod
    def save_evidence_frame(
        frame: np.ndarray, 
        output_path: str, 
        defect_type: str, 
        severity: str, 
        timestamp_sec: float, 
        gps: Tuple[float, float]
    ) -> str:
        """
        Stamps defect evidence with metadata and saves to output directory.
        """
        stamped = frame.copy()
        h, w = stamped.shape[:2]
        
        # Stamp banner on bottom
        overlay = stamped.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 18, 26), -1)
        cv2.addWeighted(overlay, 0.85, stamped, 0.15, 0, stamped)
        
        # Severity color
        col = (40, 50, 235) if severity == "CRITICAL DEFECT" else (30, 175, 245)
        cv2.rectangle(stamped, (15, h - 50), (140, h - 15), col, -1)
        cv2.putText(stamped, severity, (22, h - 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        info_str = f"DEFECT: {defect_type.upper()} | Time: {timestamp_sec:.1f}s | Lat: {gps[0]:.5f}, Lon: {gps[1]:.5f}"
        cv2.putText(stamped, info_str, (155, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
                    
        cv2.imwrite(output_path, stamped)
        return output_path

