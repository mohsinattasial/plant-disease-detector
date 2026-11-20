"""Synthetic smoke tests; not evidence of plant disease model accuracy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from train import load_dataset, features


class DatasetTests(unittest.TestCase):
    def test_missing_dataset_errors_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Dataset folder"):
                load_dataset(Path(tmp) / "missing")

    def test_synthetic_images_load_and_model_can_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, channel in (("healthy", 1), ("rust", 0)):
                folder = root / label
                folder.mkdir()
                for i in range(6):
                    img = np.zeros((56, 56, 3), dtype=np.uint8)
                    img[:, :, channel] = 180 + i
                    Image.fromarray(img).save(folder / f"sample_{i}.png")
            X, y = load_dataset(root)
            self.assertEqual(X.shape, (12, 48 * 48 * 3))
            self.assertEqual(set(y), {"healthy", "rust"})
            self.assertTrue((X >= 0).all() and (X <= 1).all())
            clf = RandomForestClassifier(n_estimators=5, random_state=42)
            clf.fit(X, y)
            self.assertEqual(len(clf.predict(X[:2])), 2)


if __name__ == "__main__":
    unittest.main()
