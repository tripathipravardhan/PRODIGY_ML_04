"""Train and evaluate a LeapGestRec hand gesture classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from gesture_utils import IMAGE_SIZE, gesture_name

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def collect_examples(data_dir: Path):
    records = []

    for subject_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            for image_path in class_dir.glob("*.png"):
                records.append(
                    (str(image_path), gesture_name(class_dir.name), subject_dir.name)
                )

    if not records:
        raise FileNotFoundError("No PNG images found. Check --data-dir.")

    return records


def make_dataset(paths, labels, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED)

    def decode(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32), label

    return dataset.map(
        decode, num_parallel_calls=AUTOTUNE
    ).batch(batch_size).prefetch(AUTOTUNE)


def build_model(class_count):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.12),
    ])

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_chart(history, path):
    plt.figure(figsize=(9, 4))

    for position, metric, title in [
        (1, "accuracy", "Accuracy"),
        (2, "loss", "Loss"),
    ]:
        plt.subplot(1, 2, position)
        plt.plot(history.history[metric], label="train")
        plt.plot(history.history[f"val_{metric}"], label="validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    tf.keras.utils.set_random_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_examples(args.data_dir)

    classes = sorted({label for _, label, _ in records})
    class_ids = {label: index for index, label in enumerate(classes)}

    train = []
    validation = []
    test = []

    # Subjects 08 and 09 are only for testing.
    # Subject 07 is only for validation.
    for path, label, subject in records:
        item = (path, class_ids[label])

        if subject in {"08", "09"}:
            test.append(item)
        elif subject == "07":
            validation.append(item)
        else:
            train.append(item)

    def unpack(items):
        paths, labels = zip(*items)
        return list(paths), list(labels)

    train_paths, train_labels = unpack(train)
    val_paths, val_labels = unpack(validation)
    test_paths, test_labels = unpack(test)

    train_ds = make_dataset(
        train_paths, train_labels, args.batch_size, training=True
    )
    val_ds = make_dataset(val_paths, val_labels, args.batch_size)
    test_ds = make_dataset(test_paths, test_labels, args.batch_size)

    model = build_model(len(classes))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                args.output_dir / "best_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
    )

    model.save(args.output_dir / "gesture_model.keras")

    (args.output_dir / "labels.json").write_text(
        json.dumps(classes, indent=2),
        encoding="utf-8",
    )

    save_chart(history, args.output_dir / "training_history.png")

    predictions = np.argmax(model.predict(test_ds), axis=1)

    loss, accuracy = model.evaluate(test_ds, verbose=0)

    report = classification_report(
        test_labels,
        predictions,
        target_names=classes,
        digits=4,
    )

    matrix = confusion_matrix(test_labels, predictions)

    result = (
        f"Test loss: {loss:.4f}\n"
        f"Test accuracy: {accuracy:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix:\n{matrix}\n"
    )

    (args.output_dir / "test_report.txt").write_text(
        result,
        encoding="utf-8",
    )

    print(result)


if __name__ == "__main__":
    main()