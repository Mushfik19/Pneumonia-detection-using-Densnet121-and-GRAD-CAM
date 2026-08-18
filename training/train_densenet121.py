"""Training script for DenseNet121 pneumonia classification.

This script is intentionally dataset-agnostic and requires user-provided data.
It does not claim or fabricate accuracy metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train DenseNet121 for pneumonia detection.')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path containing class subfolders, e.g. normal/ and pneumonia/.')
    parser.add_argument('--output-dir', type=str, default='training/artifacts', help='Where to save model and plots.')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs-frozen', type=int, default=10)
    parser.add_argument('--epochs-finetune', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--positive-class-name', type=str, default='pneumonia')
    return parser.parse_args()


def build_datasets(dataset_dir: Path, image_size: int, batch_size: int, seed: int):
    train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    class_names = train_val_ds.class_names

    val_batches = tf.data.experimental.cardinality(val_ds).numpy()
    test_size = max(1, int(0.5 * val_batches))
    test_ds = val_ds.take(test_size)
    val_ds = val_ds.skip(test_size)

    def prep(features, labels):
        features = tf.cast(features, tf.float32)
        features = preprocess_input(features)
        return features, labels

    train_ds = train_val_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def estimate_class_weights(dataset) -> dict[int, float]:
    labels = []
    for _, y_batch in dataset:
        labels.extend(y_batch.numpy().reshape(-1).tolist())

    labels = np.array(labels, dtype=np.int32)
    class_counts = np.bincount(labels)
    total = class_counts.sum()

    weights = {
        0: float(total / (2.0 * max(class_counts[0], 1))),
        1: float(total / (2.0 * max(class_counts[1], 1))),
    }
    return weights


def build_model(image_size: int, learning_rate: float):
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size, image_size, 3), name='xray_input')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='pneumonia_probability')(x)

    model = models.Model(inputs, outputs, name='densenet121_pneumonia')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    )
    return model, base_model


def plot_training_curves(history, output_dir: Path):
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(history.history.get(metric, []), label=f'train_{metric}')
        ax.plot(history.history.get(f'val_{metric}', []), label=f'val_{metric}')
        ax.set_title(metric)
        ax.legend()

    axes[-1].axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=180)
    plt.close(fig)


def evaluate_model(model, test_ds, output_dir: Path):
    y_true = []
    y_scores = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0).reshape(-1)
        y_scores.extend(preds.tolist())
        y_true.extend(y_batch.numpy().reshape(-1).tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_scores = np.array(y_scores, dtype=np.float32)
    y_pred = (y_scores >= 0.5).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'pneumonia'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=180)
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    report = classification_report(y_true, y_pred, target_names=['normal', 'pneumonia'], zero_division=0)

    metrics = {
        'auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': report,
    }

    with open(output_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(report)
    print(f"AUC: {roc_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

    train_ds, val_ds, test_ds, class_names = build_datasets(
        dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.positive_class_name not in class_names:
        raise ValueError(
            f'Positive class name {args.positive_class_name} not found in classes: {class_names}'
        )

    class_weights = estimate_class_weights(train_ds)
    model, base_model = build_model(args.image_size, args.learning_rate)

    checkpoint_path = output_dir / 'densenet121_pneumonia.keras'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
    ]

    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_frozen,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate * 0.1),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    history = history_frozen
    for key, value in history_finetune.history.items():
        history.history[key] = history.history.get(key, []) + value

    model.save(output_dir / 'densenet121_pneumonia_final.keras')
    plot_training_curves(history, output_dir)
    evaluate_model(model, test_ds, output_dir)

    print(f'Class names: {class_names}')
    print(f'Best checkpoint: {checkpoint_path}')
    print(f'Final model: {output_dir / "densenet121_pneumonia_final.keras"}')


if __name__ == '__main__':
    tf.keras.utils.set_random_seed(42)
    main()
