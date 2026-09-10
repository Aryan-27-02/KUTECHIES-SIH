import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

class AuditStore:
    """
    CSV and simple file storage manager for SafeStreet audits, defect records,
    and municipal priority rankings.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or Path(__file__).resolve().parent)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audits_csv = self.storage_dir / "audits.csv"
        self.defects_csv = self.storage_dir / "defects.csv"
        self._initialize_files()

    def _initialize_files(self):
        """Creates CSV tables with headers and seeds initial benchmark audits if empty."""
        if not self.audits_csv.exists() or os.path.getsize(self.audits_csv) == 0:
            audit_headers = [
                "audit_id", "timestamp", "school_name", "city_type", "survey_mode",
                "overall_ssi", "risk_level", "priority_score", "total_defects",
                "critical_defects", "warnings", "crossing_score", "footpath_score",
                "signage_score", "parking_score", "calming_score", "video_filename"
            ]
            with open(self.audits_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(audit_headers)
            self._seed_benchmark_audits()

        if not self.defects_csv.exists() or os.path.getsize(self.defects_csv) == 0:
            defect_headers = [
                "defect_id", "audit_id", "timestamp_sec", "gps_lat", "gps_lon",
                "param_id", "param_name", "defect_type", "severity",
                "confidence", "description", "evidence_frame_path"
            ]
            with open(self.defects_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(defect_headers)

    def _seed_benchmark_audits(self):
        """Seeds contrasting city-type audits (Tier-1 Metro vs Tier-2 Heritage/Compact)."""
        samples = [
            {
                "audit_id": "AUD-2026-001",
                "timestamp": "2026-09-08 08:30:00",
                "school_name": "St. Mary's Convent School",
                "city_type": "Tier-1 Metro (Dense Arterial)",
                "survey_mode": "Two-Wheeler",
                "overall_ssi": 42.5,
                "risk_level": "HIGH",
                "priority_score": 82.5,
                "total_defects": 14,
                "critical_defects": 6,
                "warnings": 8,
                "crossing_score": 30.0,
                "footpath_score": 25.0,
                "signage_score": 40.0,
                "parking_score": 50.0,
                "calming_score": 75.0,
                "video_filename": "st_marys_survey.mp4"
            },
            {
                "audit_id": "AUD-2026-002",
                "timestamp": "2026-09-09 13:45:00",
                "school_name": "Vidya Mandir High School",
                "city_type": "Tier-2 Compact (Mixed Residential)",
                "survey_mode": "Walking",
                "overall_ssi": 68.0,
                "risk_level": "MODERATE",
                "priority_score": 42.0,
                "total_defects": 7,
                "critical_defects": 2,
                "warnings": 5,
                "crossing_score": 60.0,
                "footpath_score": 55.0,
                "signage_score": 80.0,
                "parking_score": 70.0,
                "calming_score": 80.0,
                "video_filename": "vidya_mandir_survey.mp4"
            },
            {
                "audit_id": "AUD-2026-003",
                "timestamp": "2026-09-10 07:45:00",
                "school_name": "Greenwood Public School",
                "city_type": "Tier-1 Metro (Suburban Planned)",
                "survey_mode": "Walking",
                "overall_ssi": 86.5,
                "risk_level": "SAFE",
                "priority_score": 15.0,
                "total_defects": 2,
                "critical_defects": 0,
                "warnings": 2,
                "crossing_score": 95.0,
                "footpath_score": 90.0,
                "signage_score": 90.0,
                "parking_score": 80.0,
                "calming_score": 75.0,
                "video_filename": "greenwood_survey.mp4"
            }
        ]
        df = pd.DataFrame(samples)
        df.to_csv(self.audits_csv, index=False)

    def save_audit(self, audit_data: Dict[str, Any], defects: List[Dict[str, Any]]) -> str:
        """Saves a completed audit report and its defect items to CSV."""
        audit_id = f"AUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        row = {
            "audit_id": audit_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "school_name": audit_data.get("school_name", "Unknown School"),
            "city_type": audit_data.get("city_type", "Urban"),
            "survey_mode": audit_data.get("survey_mode", "Smartphone Walk"),
            "overall_ssi": audit_data.get("overall_ssi", 0.0),
            "risk_level": audit_data.get("risk_level", "HIGH"),
            "priority_score": audit_data.get("priority_score", 100.0),
            "total_defects": audit_data.get("total_defects", len(defects)),
            "critical_defects": audit_data.get("critical_defects", 0),
            "warnings": audit_data.get("warnings", 0),
            "crossing_score": audit_data.get("parameter_averages", {}).get("zebra_crossing", 50.0),
            "footpath_score": audit_data.get("parameter_averages", {}).get("footpath_obstruction", 50.0),
            "signage_score": audit_data.get("parameter_averages", {}).get("school_signage", 50.0),
            "parking_score": audit_data.get("parameter_averages", {}).get("obstructing_parking", 50.0),
            "calming_score": audit_data.get("parameter_averages", {}).get("speed_calming", 50.0),
            "video_filename": audit_data.get("video_filename", "uploaded_audit.mp4")
        }

        # Append to audits.csv
        df_audit = pd.DataFrame([row])
        df_audit.to_csv(self.audits_csv, mode="a", header=False, index=False)

        # Append defects to defects.csv
        if defects:
            defect_rows = []
            for idx, d in enumerate(defects):
                defect_rows.append({
                    "defect_id": f"{audit_id}-DEF-{idx+1:03d}",
                    "audit_id": audit_id,
                    "timestamp_sec": d.get("timestamp_sec", 0.0),
                    "gps_lat": d.get("gps", (0, 0))[0],
                    "gps_lon": d.get("gps", (0, 0))[1],
                    "param_id": d.get("param_id", ""),
                    "param_name": d.get("param_name", ""),
                    "defect_type": d.get("type", ""),
                    "severity": d.get("severity", "WARNING"),
                    "confidence": d.get("confidence", 0.85),
                    "description": d.get("description", ""),
                    "evidence_frame_path": d.get("evidence_frame_path", "")
                })
            df_defects = pd.DataFrame(defect_rows)
            df_defects.to_csv(self.defects_csv, mode="a", header=False, index=False)

        return audit_id

    def get_all_audits(self) -> pd.DataFrame:
        """Returns all audits sorted by municipal priority (highest urgency first)."""
        if not self.audits_csv.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.audits_csv)
        if "priority_score" in df.columns:
            df = df.sort_values(by="priority_score", ascending=False)
        return df

    def get_defects_by_audit(self, audit_id: str) -> pd.DataFrame:
        """Returns all defect entries for a specific audit."""
        if not self.defects_csv.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.defects_csv)
        return df[df["audit_id"] == audit_id]

