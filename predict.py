"""Predict a leaf class using a model trained by train.py."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.joblib"))
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    # joblib uses pickle; load only models you created/trust.
    bundle = joblib.load(args.model)
    with Image.open(args.image) as img:
        pixels = np.asarray(
            img.convert("RGB").resize(tuple(bundle["image_size"])), dtype=np.float32
        ).ravel() / 255.0
    clf = bundle["classifier"]
    label = clf.predict(pixels.reshape(1, -1))[0]
    print(f"Predicted class (unvalidated model): {label}")


if __name__ == "__main__":
    main()
