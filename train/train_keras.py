import os
import sys
from pathlib import Path
import numpy as np

# Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.generate_synthetic_data import generate_dataset

def build_model(img_size=(128, 128), num_classes=3):
    inputs = keras.Input(shape=(*img_size, 3))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    
    # Feature extraction blocks
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="condition_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SafeStreet_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_model_path = models_dir / "condition_classifier.keras"

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not train_dir.exists():
        print("[SafeStreet] Dataset not found, generating sample dataset first...")
        generate_dataset(base_dir=str(dataset_dir), samples_per_class=80)

    print("[SafeStreet] Loading dataset via tf.keras.utils.image_dataset_from_directory...")
    img_size = (128, 128)
    batch_size = 16

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("[SafeStreet] Building Keras condition classification model...")
    model = build_model(img_size=img_size, num_classes=3)
    model.summary()

    print("[SafeStreet] Training model for 8 epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        verbose=1
    )

    print(f"[SafeStreet] Saving trained model to {target_model_path}...")
    model.save(str(target_model_path))
    print("[SafeStreet] Model training complete! Successfully exported condition_classifier.keras.")

if __name__ == "__main__":
    main()

