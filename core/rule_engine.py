from typing import List, Dict, Any, Tuple
import copy

class RuleEngine:
    """
    Rule Engine: The interpretive tier of SafeStreet.
    Applies the School Zone Pedestrian Safety Rubric (IRC:67 & IRC:103)
    to translate raw computer vision detections and condition classifications
    into explicit Safety States (SAFE, WARNING, CRITICAL DEFECT).
    """

    def __init__(self):
        pass

    def evaluate_frame(
        self,
        detections: List[Dict[str, Any]],
        crossing_condition: Dict[str, Any],
        sign_condition: Dict[str, Any],
        surface_condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interprets all 5 safety parameters for a single sampled video frame.
        Returns parameter statuses, raw parameter scores (0-100), and any flagged defects.
        """
        results = {
            "parameters": {},
            "defects": []
        }

        # -------------------------------------------------------------
        # 1. PARAMETER 1: Zebra Crossing Condition
        # -------------------------------------------------------------
        cross_status = crossing_condition.get("status", "Missing / Absent (Defect)")
        if "Intact" in cross_status:
            param1_score = 100
            param1_state = "SAFE"
            param1_defect = None
        elif "Worn" in cross_status:
            param1_score = 40
            param1_state = "WARNING"
            param1_defect = {
                "param_id": "PARAM_1",
                "param_name": "Zebra Crossing Condition",
                "type": "WORN_CROSSING",
                "severity": "WARNING",
                "title": "Faded / Worn Zebra Crossing",
                "description": f"Crossing marking degraded ({crossing_condition.get('wear_percentage', 65)}% paint loss); reduced driver visibility.",
                "confidence": crossing_condition.get("confidence", 0.85)
            }
        else:
            param1_score = 10
            param1_state = "CRITICAL DEFECT"
            param1_defect = {
                "param_id": "PARAM_1",
                "param_name": "Zebra Crossing Condition",
                "type": "MISSING_CROSSING",
                "severity": "CRITICAL DEFECT",
                "title": "Missing School Crossing",
                "description": "No marked pedestrian crossing detected within active school zone gate frontage.",
                "confidence": crossing_condition.get("confidence", 0.90)
            }

        results["parameters"]["zebra_crossing"] = {
            "state": param1_state,
            "score": param1_score,
            "condition": cross_status,
            "details": crossing_condition
        }
        if param1_defect:
            results["defects"].append(param1_defect)

        # -------------------------------------------------------------
        # 2. PARAMETER 2: Footpath Walkability & Encroachment
        # -------------------------------------------------------------
        # Count vehicles parked on footpath zones
        footpath_vehicles = [
            d for d in detections 
            if d.get("zone") in ["footpath_left", "footpath_right"] 
            and d.get("class") in ["motorcycle", "car", "bicycle", "truck", "bus"]
        ]
        pedestrians_in_road = [
            d for d in detections 
            if d.get("zone") == "carriageway" and d.get("class") == "person"
        ]

        num_encroachments = len(footpath_vehicles)
        if num_encroachments == 0:
            param2_score = 100
            param2_state = "SAFE"
            param2_defect = None
        elif num_encroachments == 1:
            param2_score = 50
            param2_state = "WARNING"
            param2_defect = {
                "param_id": "PARAM_2",
                "param_name": "Footpath Walkability",
                "type": "FOOTPATH_ENCROACHMENT",
                "severity": "WARNING",
                "title": "Partial Footpath Encroachment",
                "description": f"{num_encroachments} vehicle parked on pedestrian walkway, narrowing safe walking corridor.",
                "confidence": 0.88,
                "bbox": footpath_vehicles[0]["bbox"]
            }
        else:
            param2_score = 15
            param2_state = "CRITICAL DEFECT"
            param2_defect = {
                "param_id": "PARAM_2",
                "param_name": "Footpath Walkability",
                "type": "FOOTPATH_BLOCKED",
                "severity": "CRITICAL DEFECT",
                "title": "Severe Footpath Blockage",
                "description": f"{num_encroachments} vehicles parked on pedestrian path; forces children into active vehicular carriageway.",
                "confidence": 0.92,
                "bbox": footpath_vehicles[0]["bbox"]
            }

        results["parameters"]["footpath_obstruction"] = {
            "state": param2_state,
            "score": param2_score,
            "encroaching_vehicles_count": num_encroachments,
            "pedestrians_in_road_count": len(pedestrians_in_road)
        }
        if param2_defect:
            results["defects"].append(param2_defect)

        # -------------------------------------------------------------
        # 3. PARAMETER 3: School Zone Signage Presence & Visibility
        # -------------------------------------------------------------
        sign_status = sign_condition.get("status", "Missing Sign (Defect)")
        if "Clearly Visible" in sign_status:
            param3_score = 100
            param3_state = "SAFE"
            param3_defect = None
        elif "Damaged" in sign_status or "Obscured" in sign_status:
            param3_score = 45
            param3_state = "WARNING"
            param3_defect = {
                "param_id": "PARAM_3",
                "param_name": "School Zone Signage",
                "type": "OBSCURED_SIGNAGE",
                "severity": "WARNING",
                "title": "Obscured / Substandard Signage",
                "description": "School cautionary sign is obscured by foliage, faded, or damaged.",
                "confidence": sign_condition.get("confidence", 0.82)
            }
        else:
            param3_score = 10
            param3_state = "CRITICAL DEFECT"
            param3_defect = {
                "param_id": "PARAM_3",
                "param_name": "School Zone Signage",
                "type": "MISSING_SIGNAGE",
                "severity": "CRITICAL DEFECT",
                "title": "Missing School Zone Signage",
                "description": "Mandatory IRC:67 School Warning / Speed Restriction sign absent on school approach.",
                "confidence": sign_condition.get("confidence", 0.90)
            }

        results["parameters"]["school_signage"] = {
            "state": param3_state,
            "score": param3_score,
            "condition": sign_status
        }
        if param3_defect:
            results["defects"].append(param3_defect)

        # -------------------------------------------------------------
        # 4. PARAMETER 4: Blind-Spot Obstructing Parking
        # -------------------------------------------------------------
        # Vehicles positioned in crossing approach or directly near pedestrians
        cars_in_approach = [
            d for d in detections 
            if d.get("zone") == "crossing_approach" 
            and d.get("class") in ["car", "bus", "truck"]
        ]
        
        if len(cars_in_approach) == 0:
            param4_score = 100
            param4_state = "SAFE"
            param4_defect = None
        elif len(cars_in_approach) == 1:
            param4_score = 55
            param4_state = "WARNING"
            param4_defect = {
                "param_id": "PARAM_4",
                "param_name": "Blind-Spot Parking",
                "type": "RESTRICTED_PARKING",
                "severity": "WARNING",
                "title": "Dispersal Buffer Parking",
                "description": "Vehicle stationary near crossing zone; creates partial sightline restriction.",
                "confidence": 0.85,
                "bbox": cars_in_approach[0]["bbox"]
            }
        else:
            param4_score = 20
            param4_state = "CRITICAL DEFECT"
            param4_defect = {
                "param_id": "PARAM_4",
                "param_name": "Blind-Spot Parking",
                "type": "BLIND_SPOT_OBSTRUCTION",
                "severity": "CRITICAL DEFECT",
                "title": "Critical Blind Spot Hazard",
                "description": "Multiple vehicles parked in school dispersal buffer; driver cannot see emerging children.",
                "confidence": 0.91,
                "bbox": cars_in_approach[0]["bbox"]
            }

        results["parameters"]["obstructing_parking"] = {
            "state": param4_state,
            "score": param4_score,
            "obstructing_vehicles_count": len(cars_in_approach)
        }
        if param4_defect:
            results["defects"].append(param4_defect)

        # -------------------------------------------------------------
        # 5. PARAMETER 5: Speed Calming & Surface Integrity
        # -------------------------------------------------------------
        surf_status = surface_condition.get("status", "Good Condition / Calmed")
        if "Good" in surf_status:
            param5_score = 100
            param5_state = "SAFE"
            param5_defect = None
        else:
            param5_score = 30
            param5_state = "CRITICAL DEFECT"
            param5_defect = {
                "param_id": "PARAM_5",
                "param_name": "Speed Calming & Surface",
                "type": "SURFACE_CALMING_DEFECT",
                "severity": "CRITICAL DEFECT",
                "title": "Missing Speed Calming / Surface Hazard",
                "description": "Absence of raised speed calming table / severe road potholes inducing vehicle hazard.",
                "confidence": surface_condition.get("confidence", 0.84)
            }

        results["parameters"]["speed_calming"] = {
            "state": param5_state,
            "score": param5_score,
            "condition": surf_status
        }
        if param5_defect:
            results["defects"].append(param5_defect)

        return results

