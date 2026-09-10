import cv2
import numpy as np
from pathlib import Path

def generate_school_zone_video(output_path="demo_assets/sample_school_zone.mp4", duration_sec=8, fps=20):
    """
    Synthesizes a realistic smartphone perspective video recorded from a walking/two-wheeler survey
    approaching an urban school zone. Features:
    - Roadway with sidewalk buffers
    - Faded zebra crossing stripes
    - Two-wheelers parked on the sidewalk (encroachment defect)
    - School gate signage
    - Exiting pedestrians / children
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    w, h = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    total_frames = duration_sec * fps

    for f in range(total_frames):
        # Base scene: Road perspective
        frame = np.full((h, w, 3), (70, 75, 80), dtype=np.uint8)

        # Draw Sidewalks (pavement) on left and right
        cv2.rectangle(frame, (0, 0), (int(w * 0.22), h), (130, 135, 140), -1)
        cv2.rectangle(frame, (int(w * 0.78), 0), (w, h), (130, 135, 140), -1)

        # Curb lines (yellow & black curb blocks)
        for y in range(0, h, 40):
            color = (30, 200, 240) if (y // 40) % 2 == 0 else (30, 30, 30)
            cv2.rectangle(frame, (int(w * 0.21), y), (int(w * 0.22), y + 40), color, -1)
            cv2.rectangle(frame, (int(w * 0.78), y), (int(w * 0.79), y + 40), color, -1)

        # Approaching perspective motion offset
        progress = f / float(total_frames)
        y_offset = int(progress * 180)

        # 1. Parameter 1: Zebra Crossing markings (approaching in view)
        crossing_y = int(h * 0.45) + y_offset
        if crossing_y < h:
            stripe_h = 35
            for sx in range(int(w * 0.24), int(w * 0.76), 55):
                # Make stripes look degraded/worn (faded white with missing patches)
                cv2.rectangle(frame, (sx, crossing_y), (sx + 32, crossing_y + stripe_h), (180, 185, 190), -1)
                # Chipped/worn spots
                cv2.circle(frame, (sx + 15, crossing_y + 18), 7, (70, 75, 80), -1)

        # 2. Parameter 2: Parked Two-Wheelers blocking the footpath (Left Sidewalk)
        # Represents two-wheelers encroaching on pedestrian walkway
        bike1_y = int(h * 0.4) + int(y_offset * 0.8)
        if bike1_y < h - 50:
            # Bike 1 (encroaching)
            cv2.rectangle(frame, (35, bike1_y), (105, bike1_y + 60), (30, 110, 210), -1)
            cv2.circle(frame, (50, bike1_y + 55), 12, (20, 20, 20), -1)
            cv2.circle(frame, (90, bike1_y + 55), 12, (20, 20, 20), -1)
            cv2.putText(frame, "TWO-WHEELER", (35, bike1_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Bike 2 (encroaching)
            cv2.rectangle(frame, (45, bike1_y + 80), (115, bike1_y + 140), (180, 40, 40), -1)
            cv2.circle(frame, (60, bike1_y + 135), 12, (20, 20, 20), -1)
            cv2.circle(frame, (100, bike1_y + 135), 12, (20, 20, 20), -1)

        # 3. Parameter 3: School Gate Signage (Right side, high up)
        sign_y = int(h * 0.15)
        # Yellow triangular caution sign
        pts = np.array([[int(w * 0.86), sign_y], [int(w * 0.81), sign_y + 70], [int(w * 0.91), sign_y + 70]], np.int32)
        cv2.fillPoly(frame, [pts], (30, 210, 245))
        cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
        cv2.putText(frame, "SCHOOL", (int(w * 0.825), sign_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 4. Parameter 4: Car parked on right edge (blind spot)
        car_y = int(h * 0.3) + int(y_offset * 0.6)
        if car_y < h - 90:
            cv2.rectangle(frame, (int(w * 0.68), car_y), (int(w * 0.77), car_y + 85), (140, 70, 70), -1)
            cv2.putText(frame, "PARKED CAR", (int(w * 0.68), car_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Pedestrians / Children walking
        ped_y = int(h * 0.5) + int(y_offset * 0.7)
        if ped_y < h - 40:
            # Child walking near road edge because sidewalk is blocked
            cv2.circle(frame, (int(w * 0.25), ped_y), 9, (220, 190, 160), -1)
            cv2.rectangle(frame, (int(w * 0.25) - 7, ped_y + 9), (int(w * 0.25) + 7, ped_y + 35), (40, 180, 70), -1)
            cv2.putText(frame, "STUDENT", (int(w * 0.25) - 15, ped_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Add camera motion vibration / sway
        shake_x = np.random.randint(-2, 3)
        shake_y = np.random.randint(-2, 3)
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        frame = cv2.warpAffine(frame, M, (w, h))

        out.write(frame)

    out.release()
    print(f"[SafeStreet] Sample school zone demo video generated at {out_file}")
    return str(out_file)

if __name__ == "__main__":
    generate_school_zone_video()

