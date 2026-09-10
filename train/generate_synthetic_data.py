import os
import cv2
import numpy as np
from pathlib import Path

def create_road_texture(w=128, h=128):
    """Creates realistic asphalt base with subtle noise."""
    base = np.random.randint(45, 65, (h, w, 3), dtype=np.uint8)
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    asphalt = np.clip(base.astype(np.int16) + noise, 20, 90).astype(np.uint8)
    return asphalt

def generate_dataset(base_dir="dataset", samples_per_class=60):
    """
    Generates annotated training and validation dataset for road condition classification:
    - 0_intact: High-contrast, well-painted zebra crossing stripes
    - 1_worn: Eroded, degraded, faded crossing stripes
    - 2_missing: Unpainted asphalt with no pedestrian crossing
    """
    base_path = Path(base_dir)
    splits = {
        "train": int(samples_per_class * 0.8),
        "val": int(samples_per_class * 0.2)
    }

    classes = ["0_intact", "1_worn", "2_missing"]

    for split_name, count in splits.items():
        for cls_name in classes:
            target_dir = base_path / split_name / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                img = create_road_texture(128, 128)
                
                if cls_name == "0_intact":
                    # Crisp white stripes
                    stripe_width = 18
                    for x in range(10, 128, 30):
                        cv2.rectangle(img, (x, 15), (x + stripe_width, 115), (230, 235, 240), -1)
                        # Add slight realistic wear
                        if np.random.rand() > 0.5:
                            dust = np.random.randint(0, 35, (100, stripe_width, 3), dtype=np.uint8)
                            img[15:115, x:x+stripe_width] = np.clip(
                                img[15:115, x:x+stripe_width].astype(np.int16) - dust.astype(np.int16), 0, 255
                            ).astype(np.uint8)

                elif cls_name == "1_worn":
                    # Heavily faded, chipped stripes
                    stripe_width = 18
                    for x in range(10, 128, 30):
                        # Faded greyish-white
                        cv2.rectangle(img, (x, 15), (x + stripe_width, 115), (140, 145, 150), -1)
                        # Simulate heavy erosion / potholes / flaked paint
                        for _ in range(15):
                            cx = np.random.randint(x, x + stripe_width)
                            cy = np.random.randint(15, 115)
                            r = np.random.randint(2, 6)
                            cv2.circle(img, (cx, cy), r, (50, 55, 60), -1)

                elif cls_name == "2_missing":
                    # Plain asphalt, maybe random crack or tire track
                    if np.random.rand() > 0.4:
                        pt1 = (np.random.randint(0, 40), np.random.randint(0, 128))
                        pt2 = (np.random.randint(80, 128), np.random.randint(0, 128))
                        cv2.line(img, pt1, pt2, (25, 28, 32), 2)

                filename = target_dir / f"sample_{i:04d}.jpg"
                cv2.imwrite(str(filename), img)

    print(f"[SafeStreet] Successfully generated synthetic condition dataset in {base_path}")

if __name__ == "__main__":
    generate_dataset()

