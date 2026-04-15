# ============================================================
# models/cnn_model.py
# CNN architecture for mel-spectrogram binary classification.
#
# ARCHITECTURE RATIONALE
# ──────────────────────
# Input  : (128, 128, 1)  — grayscale mel spectrogram
# Output : scalar in (0,1) — P(Faulty)
#
# Block structure:
#   Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
#   ×3 with increasing filter counts (32 → 64 → 128)
#
# Why increasing filters?
#   Early layers detect low-level patterns (sharp edges = transients,
#   horizontal bands = harmonic peaks).  Deeper layers combine those
#   into higher-level fault signatures.
#
# Why Batch Normalisation?
#   Stabilises training by normalising each mini-batch's activations,
#   allowing higher learning rates and reducing sensitivity to init.
#
# Why GlobalAveragePooling instead of Flatten?
#   GAP collapses spatial dimensions to one value per feature map,
#   drastically reducing parameters and acting as a built-in regulariser.
#   It also makes the model tolerant of slight input size changes.
#
# Loss : Binary Cross-Entropy  (standard for 2-class problems with
#                                sigmoid output)
# Opt  : Adam  (adaptive LR per parameter — robust, low-tuning)
# ============================================================

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

from config.settings import model_cfg
from utils.logger import logger


def build_cnn(
    input_shape: Tuple[int, int, int] = model_cfg.input_shape,
    dropout_rate: float = model_cfg.dropout_rate,
    learning_rate: float = model_cfg.learning_rate,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Build and compile the CNN classifier.

    Args:
        input_shape:    (height, width, channels) — default (128, 128, 1).
        dropout_rate:   Fraction of neurons dropped during training.
        learning_rate:  Adam initial learning rate.
        l2_reg:         L2 weight regularisation coefficient.

    Returns:
        Compiled tf.keras.Model ready for .fit().
    """

    def conv_block(
        x: tf.Tensor,
        filters: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = dropout_rate,
    ) -> tf.Tensor:
        """
        Double-Conv → BN → MaxPool → Dropout block.
        Stacking two convolutions before pooling lets the network build
        richer representations without additional pooling stages.
        """
        # First conv
        x = layers.Conv2D(
            filters, kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,   # BN provides bias implicitly
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Second conv (same filters — deepens without widening)
        x = layers.Conv2D(
            filters, kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(pool_size)(x)
        x = layers.Dropout(dropout)(x)
        return x

    # ── Input ────────────────────────────────────────────────────────────
    inp = layers.Input(shape=input_shape, name="spectrogram_input")

    # ── Convolutional backbone ───────────────────────────────────────────
    x = conv_block(inp, filters=32,  pool_size=2, dropout=dropout_rate * 0.75)  # 128→64
    x = conv_block(x,   filters=64,  pool_size=2, dropout=dropout_rate * 0.75)  # 64→32
    x = conv_block(x,   filters=128, pool_size=2, dropout=dropout_rate)          # 32→16

    # Extra conv layer for richer feature extraction (no pooling)
    x = layers.Conv2D(
        256, 3, padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # ── Head ─────────────────────────────────────────────────────────────
    # GlobalAveragePooling — one scalar per feature map → (batch, 256)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name=model_cfg.model_name)

    # ── Compile ──────────────────────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info(f"Model '{model.name}' built — params: {model.count_params():,}")
    return model


def load_model(path: str) -> tf.keras.Model:
    """
    Load a saved Keras model from disk.

    Args:
        path: Full path to the .h5 or SavedModel directory.

    Returns:
        Compiled tf.keras.Model.
    """
    model = tf.keras.models.load_model(path)
    logger.info(f"Model loaded from '{path}'")
    return model
