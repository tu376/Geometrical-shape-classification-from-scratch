# Geometrical Shape Classification from Scratch

## Project Overview

This repository implements a simple convolutional neural network (CNN) from scratch for classifying geometric shape images. The project includes:

- a custom CNN implementation using plain NumPy and manual layer classes
- training and validation code for shape classification
- a baseline suite that evaluates classical machine learning models on raw pixel and CNN feature representations
- image augmentation utilities and visualization tools for training curves, confusion matrices, and predictions

## Repository Structure

- `Project I/train.py` — main training script for the CNN model
- `Project I/model.py` — CNN model architecture and weight save/load methods
- `Project I/baseline.py` — baseline experiments using KNN, SVM, and Random Forest
- `Project I/utils/` — helper modules for data loading, augmentation, metrics, and visualization
- `Project I/layers/` — custom neural network layer implementations
- `Project I/dataset/` — image dataset folders for training/validation and test data
- `Project I/labels.csv` — labels for the training/validation dataset
- `Project I/test.csv` — labels for the test dataset
- `Project I/weights.npy` — saved model weights (if available)

## Supported Shape Classes

The model is trained to classify the following shapes:

- circle
- ellipse
- square
- triangle
- rectangle
- hexagon
- octagon

## Requirements

This project is written in Python and depends on the following packages:

- `numpy`
- `Pillow`
- `scipy`
- `matplotlib`
- `scikit-learn`

You can install them with:

```bash
pip install numpy pillow scipy matplotlib scikit-learn
```

## How to Run

Open a terminal in the project root and run the main training script from the `Project I` folder:

```bash
cd "Project I"
python train.py
```

The script will:

- load training and validation data from `dataset/train_valid`
- split the data into training and validation sets
- train the CNN model for a fixed number of epochs
- save the best weights to `weights.npy`
- evaluate the model on validation data
- optionally evaluate on the test set if `test.csv` and `dataset/test` are present
- prompt for a new image path to run a final prediction

## Baseline Experiments

To compare the CNN features with classical methods, run:

```bash
cd "Project I"
python baseline.py
```

This script extracts features from:

- raw pixel values
- the CNN feature extractor before the final linear layer

Then it trains and evaluates the following classifiers:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Results are printed with accuracy scores and a confusion matrix image.

## Output Files

During execution, the project may generate:

- `training_curves.png` — training and validation loss/accuracy plots
- `confusion_matrix_test.png` — confusion matrix for the test set
- `confusion_matrix_svm_cnn.png` — confusion matrix for the baseline SVM model using CNN features

## Notes

- Images are resized to 64×64 grayscale before being processed.
- The dataset CSV files should contain `filename` and `label` columns.
- The code uses simple custom layers instead of a deep learning framework.

## License

This repository is provided as a learning example for shape classification and neural network implementation from scratch.
