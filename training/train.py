# ============================================================
# training/train.py
# End-to-end training pipeline with:
#   • Keras callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
#   • History logging
#   • Full evaluation on the held-out test set
#   • Confusion matrix + classification report
#
# Run directly:
#   python training/train.py
# ============================================================

import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # headless backend — safe for servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import model_cfg, data_cfg
from models.cnn_model import build_cnn
from training.data_pipeline import build_dataset, split_dataset
from utils.logger import logger
from utils.helpers import ensure_dir


# ── Callback factory ─────────────────────────────────────────────────────────

def make_callbacks(checkpoint_path: Path) -> list:
    """
    Standard production callback stack:
      - EarlyStopping     : stop when val_loss stagnates
      - ReduceLROnPlateau : halve LR on plateau before stopping
      - ModelCheckpoint   : save best weights automatically
      - TensorBoard       : optional — uncomment if you want live graphs
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=model_cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_history(history: tf.keras.callbacks.History, save_dir: Path) -> None:
    """Save training curves (loss + accuracy) as PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Training curves saved → {out}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
) -> None:
    """Save confusion matrix plot as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Faulty"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    out = save_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out}")


# ── Main training entry point ─────────────────────────────────────────────────

def train(
    epochs: int        = model_cfg.epochs,
    batch_size: int    = model_cfg.batch_size,
    learning_rate: float = model_cfg.learning_rate,
) -> None:
    """
    Full training pipeline:
      1. Build feature dataset from WAV files
      2. Split into train / val / test
      3. Build CNN model
      4. Train with callbacks
      5. Evaluate on test set
      6. Save model + metrics
    """
    save_dir = ensure_dir(data_cfg.saved_models_dir)
    logs_dir = ensure_dir(data_cfg.logs_dir)

    # ── 1. Data ──────────────────────────────────────────────────────────
    logger.info("Building feature dataset …")
    X, y = build_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    logger.info(
        f"Shapes — train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}"
    )

    # ── 2. Model ─────────────────────────────────────────────────────────
    model = build_cnn(learning_rate=learning_rate)
    model.summary(print_fn=logger.debug)

    # ── 3. Train ─────────────────────────────────────────────────────────
    checkpoint_path = save_dir / f"{model_cfg.model_name}.h5"
    callbacks = make_callbacks(checkpoint_path)

    logger.info(f"Training for up to {epochs} epochs …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    _plot_history(history, logs_dir)

    # ── 4. Evaluate on test set ──────────────────────────────────────────
    logger.info("Evaluating on held-out test set …")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_metrics = dict(zip(model.metrics_names, test_results))
    logger.info(f"Test metrics: {test_metrics}")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Faulty"],
        output_dict=True,
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Faulty"]))

    _plot_confusion_matrix(y_test, y_pred, logs_dir)

    # ── 5. Save metrics JSON ─────────────────────────────────────────────
    metrics_payload = {
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "classification_report": report,
        "roc_auc": roc_auc,
        "epochs_trained": len(history.history["loss"]),
    }
    metrics_path = logs_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    logger.success(
        f"Training complete.  Best model → {checkpoint_path}\n"
        f"  Test accuracy : {test_metrics.get('accuracy', 0):.4f}\n"
        f"  Test AUC      : {roc_auc:.4f}"
    )


if __name__ == "__main__":
    train()
