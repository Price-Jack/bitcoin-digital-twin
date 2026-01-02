"""Model training utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


def train_return_model(x_train_scaled: np.ndarray, y_return_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train_scaled, y_return_train)
    return model


def train_direction_model(x_train_scaled: np.ndarray, y_direction_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000)
    model.fit(x_train_scaled, y_direction_train)
    return model


def build_lstm_classifier(lookback: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm_classifier(
    x_seq_train: np.ndarray,
    y_seq_train: np.ndarray,
    lookback: int,
    n_features: int,
    epochs: int = 50,
    batch_size: int = 64,
    patience: int = 6,
    validation_split: float = 0.2,
    verbose: int = 1,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, dict]:
    """Train LSTM with balanced class weights (matches notebook)."""
    tf.keras.backend.clear_session()

    cls_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_seq_train),
        y=y_seq_train,
    )
    class_weights = {0: cls_weights[0], 1: cls_weights[1]}

    model = build_lstm_classifier(lookback=lookback, n_features=n_features)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        x_seq_train,
        y_seq_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=verbose,
    )
    return model, history, class_weights
