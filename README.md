# YOLO Self-Supervised Training Pipeline

This repository provides a **self-supervised training pipeline** for pathology image detection using **Ultralytics YOLO**.  
It supports patch preparation from whole-slide images, dataset splitting, and multi-generation pseudo-label training with configurable hyperparameters.

---

## 📂 Project Structure

- **`prepare_data.py`**  
  Prepares the dataset by cutting patches from slide images and exporting them in YOLO format. Handles bounding box matching, class mapping, and train/val split.

- **`slide_data.py`**  
  Core utilities for handling slide data:  
  - `SlideData`: loads individual SVS slides and annotations, creates patches, matches boxes, exports YOLO-ready images/labels.  
  - `SlideDataList`: wraps multiple slides, builds global class maps, exports patches, creates manifests.

- **`train_model_selfsuper.py`**  
  Standard **multi-generation pseudo-label training script**.  
  Iteratively:
  1. Trains YOLO with given configuration.  
  2. Predicts pseudo-labels.  
  3. Merges with ground-truth labels.  
  4. Re-trains on the merged dataset.  
  Tracks metrics, plots class counts, and manages runs.

- **`train_model_selfsuper_random_gen.py`**  
  Variant of the above, but introduces **randomized seeds and validation thresholds** to conduct multiple experiments.  
  Useful for robustness testing and hyperparameter search.

- **`config.json`**  
  Base configuration file (deterministic).  
  Includes dataset paths, YOLO training parameters, confidence decay schedule, number of generations, augmentation settings, and validation thresholds.

- **`config_random_gen.json`**  
  Variant config tuned for **random-seed experiments**.  
  Similar to `config.json` but allows randomized seeds and validation confidence for multiple runs.

---

## ⚙️ Installation

1. Clone this repository and install dependencies:
   ```bash
   git clone <your_repo_url>
   cd <your_repo>
   pip install -r requirements.txt
