# Plant Leaf Image Classifier — Starter Project

A small, reproducible **educational baseline**, using image features and a RandomForest classifier. This repository previously had no source files; it now provides an explicit starter implementation.

## Prepare labeled data
Place photos in folders named after their labels:

```text
data/
  healthy/
    leaf1.jpg
    leaf2.jpg
  rust/
    leaf1.jpg
    leaf2.jpg
  blight/
    leaf1.jpg
    leaf2.jpg
```

Use **at least several dozen distinct photos per class**. Never put copies of the same leaf in both train and test sets.

## Set up and train
```bash
python -m pip install -r requirements.txt
python train.py --data data --model model.joblib
python predict.py --model model.joblib --image path/to/leaf.jpg
```

Training prints a held-out test classification report and saves the fitted model. Data, trained model and evaluation results are **not included**. Predictions cannot be trusted without representative, independent testing.

This baseline uses resized RGB pixel vectors (not a deep neural network). It is not a validated diagnosis tool. Good next steps: class balancing, cross-validation, leakage checks and a pretrained image model with real field validation.

Author: Mohsin Atta.
