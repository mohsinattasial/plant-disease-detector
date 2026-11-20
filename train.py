"""Train a simple leaf image classification baseline from labeled folders."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = (48, 48)


def features(path: Path) -> np.ndarray:
    """Return a normalized fixed-length RGB vector."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32).ravel() / 255.0


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    samples, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                samples.append(features(path))
                labels.append(folder.name)
            except (OSError, UnidentifiedImageError) as exc:
                print(f"Skipping unreadable image {path}: {exc}")
    if len(set(labels)) < 2:
        raise ValueError("Provide at least two labeled class folders with valid images.")
    counts = {label: labels.count(label) for label in set(labels)}
    if any(count < 5 for count in counts.values()):
        raise ValueError(f"Provide at least five images per class for a stratified split: {counts}")
    return np.stack(samples), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("model.joblib"))
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "image_size": IMAGE_SIZE}, args.model)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
