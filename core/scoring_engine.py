from typing import List, Dict, Any, Tuple
import numpy as np
from config import SAFETY_PARAMETERS, RISK_LEVELS

class ScoringEngine:
    """
    Scoring Engine: The evaluation tier of SafeStreet.
    Calculates the 5-parameter School Safety Index (SSI: 0-100),
    determines the overall risk level, aggregates whole-audit metrics,
    and computes the Municipal Intervention Priority Rank.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {k: v["weight"] for k, v in SAFETY_PARAMETERS.items()}

    def calculate_frame_score(self, parameters: Dict[str, Dict[str, Any]]) -> Tuple[float, str, Dict[str, float]]:
        """
        Computes composite School Safety Index (SSI) for a single frame.
        Returns (ssi_score, risk_level, weighted_breakdown)
        """
        weighted_breakdown = {}
        total_score = 0.0

        for param_key, param_info in parameters.items():
            weight = self.weights.get(param_key, 0.20)
            score = param_info.get("score", 100)
            weighted_val = score * weight
            weighted_breakdown[param_key] = round(weighted_val, 2)
            total_score += weighted_val

        total_score = round(max(0.0, min(100.0, total_score)), 1)
        risk_level = self.determine_risk_level(total_score)
        
        return total_score, risk_level, weighted_breakdown

    def determine_risk_level(self, score: float) -> str:
        """Categorizes score into SAFE, MODERATE, or HIGH risk."""
        if score >= RISK_LEVELS["SAFE"]["min_score"]:
            return "SAFE"
        elif score >= RISK_LEVELS["MODERATE"]["min_score"]:
            return "MODERATE"
        else:
            return "HIGH"

    def aggregate_audit(
        self,
        school_name: str,
        city_type: str,
        frame_evaluations: List[Dict[str, Any]],
        all_defects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates frame-level evaluations into a comprehensive School Zone Audit.
        Computes Municipal Priority Score, breakdown by parameter, and selects key evidence.
        """
        if not frame_evaluations:
            return {
                "school_name": school_name,
                "city_type": city_type,
                "overall_ssi": 0.0,
                "risk_level": "HIGH",
                "priority_score": 100.0,
                "total_defects": 0,
                "critical_defects": 0,
                "warnings": 0,
                "parameter_averages": {},
                "summary": "No frames evaluated."
            }

        # Calculate average score per parameter
        param_sums = {k: 0.0 for k in SAFETY_PARAMETERS}
        frame_scores = []

        for eval_item in frame_evaluations:
            params = eval_item["parameters"]
            frame_ssi, _, _ = self.calculate_frame_score(params)
            frame_scores.append(frame_ssi)
            for k in param_sums:
                param_sums[k] += params.get(k, {}).get("score", 100)

        num_frames = len(frame_evaluations)
        param_averages = {k: round(v / num_frames, 1) for k, v in param_sums.items()}

        # Overall composite SSI
        overall_ssi = round(sum(param_averages[k] * self.weights[k] for k in self.weights), 1)
        risk_level = self.determine_risk_level(overall_ssi)

        # Defect analysis
        critical_count = sum(1 for d in all_defects if d.get("severity") == "CRITICAL DEFECT")
        warning_count = sum(1 for d in all_defects if d.get("severity") == "WARNING")

        # Municipal Priority Score (0-100, where 100 = highest urgency for funding & repair)
        # Priority increases with lower SSI and higher critical defect density
        urgency_penalty = min(25.0, critical_count * 2.5)
        priority_score = round(min(100.0, max(0.0, (100.0 - overall_ssi) + urgency_penalty)), 1)

        return {
            "school_name": school_name,
            "city_type": city_type,
            "overall_ssi": overall_ssi,
            "risk_level": risk_level,
            "priority_score": priority_score,
            "total_defects": len(all_defects),
            "critical_defects": critical_count,
            "warnings": warning_count,
            "parameter_averages": param_averages,
            "frames_analyzed": num_frames,
            "status_badge": RISK_LEVELS[risk_level]["badge"],
            "status_color": RISK_LEVELS[risk_level]["color"]
        }

