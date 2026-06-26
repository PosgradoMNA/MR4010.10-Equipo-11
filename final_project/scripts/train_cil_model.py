import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "driving_log.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cil_model.h5")

IMG_WIDTH = 320
IMG_HEIGHT = 160
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
NUM_COMMANDS = 4

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.15


def load_dataset():
    """Load CSV and return dataframe."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from CSV")
    print(f"Command distribution:\n{df['command'].value_counts().sort_index()}")
    return df


def load_image(img_name):
    """Load and normalize a single image."""
    path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def augment_flip(image, steering):
    """Horizontal flip + negate steering."""
    return cv2.flip(image, 1), -steering


def augment_brightness(image):
    """Random brightness adjustment."""
    factor = np.random.uniform(0.6, 1.4)
    return np.clip(image * factor, 0.0, 1.0).astype(np.float32)


def build_augmented_dataset(df):
    """Load all images and apply augmentation. Returns images, steerings, commands."""
    images = []
    steerings = []
    commands = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"  Loading images: {idx}/{total}")

        img = load_image(row['image_name'])
        if img is None:
            continue

        steering = float(row['steering_angle'])
        cmd = int(row['command'])

        images.append(img)
        steerings.append(steering)
        commands.append(cmd)

        flipped_img, flipped_steer = augment_flip(img, steering)
        flipped_cmd = cmd
        if cmd == 1:
            flipped_cmd = 2
        elif cmd == 2:
            flipped_cmd = 1
        images.append(flipped_img)
        steerings.append(flipped_steer)
        commands.append(flipped_cmd)

        bright_img = augment_brightness(img)
        images.append(bright_img)
        steerings.append(steering)
        commands.append(cmd)

    images = np.array(images)
    steerings = np.array(steerings, dtype=np.float32)
    commands = np.array(commands, dtype=np.int32)

    print(f"  Dataset after augmentation: {len(images)} samples")
    return images, steerings, commands


def build_model():
    """
    CNN backbone + 4 branched dense heads.
    Input: image + command (one-hot)
    Output: steering angle
    """
    img_input = layers.Input(shape=IMG_SHAPE, name="image")

    x = layers.Conv2D(32, 5, strides=2, activation='relu')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, strides=2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    features = layers.Dense(256, activation='relu', name="features")(x)

    cmd_input = layers.Input(shape=(NUM_COMMANDS,), name="command")

    branches = []
    for i in range(NUM_COMMANDS):
        branch = layers.Dense(128, activation='relu')(features)
        branch = layers.Dropout(0.3)(branch)
        branch = layers.Dense(64, activation='relu')(branch)
        branch = layers.Dense(1, activation='tanh', name=f"branch_{i}")(branch)
        branches.append(branch)

    stacked = layers.Concatenate(axis=-1)(branches)
    output = layers.Dot(axes=1, name="steering")([stacked, cmd_input])

    model = keras.Model(inputs=[img_input, cmd_input], outputs=output)
    return model


def main():
    print("=" * 60)
    print("  CIL MODEL TRAINING")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    df = load_dataset()

    print("\n[2/4] Building augmented dataset...")
    images, steerings, commands = build_augmented_dataset(df)

    max_steer = 0.5
    steerings = steerings / max_steer
    steerings = np.clip(steerings, -1.0, 1.0)

    commands_onehot = np.eye(NUM_COMMANDS)[commands]

    (train_imgs, val_imgs, train_steer, val_steer,
     train_cmd, val_cmd) = train_test_split(
        images, steerings, commands_onehot,
        test_size=VALIDATION_SPLIT, random_state=42
    )

    print(f"\n  Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    print("\n[3/4] Building model...")
    model = build_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    print("\n[4/4] Training...")
    history = model.fit(
        {"image": train_imgs, "command": train_cmd},
        train_steer,
        validation_data=({"image": val_imgs, "command": val_cmd}, val_steer),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  Best val_mae: {min(history.history['val_mae']):.6f}")


if __name__ == "__main__":
    main()
