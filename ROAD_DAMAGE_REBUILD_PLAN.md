# Road Damage Detection Rebuild Plan

## Project Goal
Rebuild `RoadDamage` into a Python 3.14-compatible road damage and pothole detection system using YOLO-based object detection, trained on the RDD2022 China Drone dataset.

## Current State
- Current notebook uses TensorFlow/Keras models and legacy YOLO-style code that is not compatible with Python 3.14.
- Existing dataset annotations are in Pascal VOC XML format, and sample images are available under `RDD2022_China_Drone`.
- A working report exists describing YOLOv5, YOLOv7, YOLOv8, dataset preprocessing, and performance evaluation.
- Current `requirements.txt` still includes TensorFlow, which blocks Python 3.14 compatibility.

## Target Architecture
1. Data ingestion and annotation conversion
   - Parse XML annotations and convert to YOLO-format labels.
   - Create a `data.yaml` dataset config for ultralytics training.
2. Model training and evaluation
   - Use `ultralytics` YOLOv8 or YOLOv9 for training on the RDD2022 dataset.
   - Train a lightweight model such as `yolov8n` or `yolov8m` for faster execution and compatibility.
3. Inference and visualization
   - Provide a clean inference pipeline for damage detection on sample images.
   - Display detection boxes and class labels with OpenCV/Matplotlib.
4. Reporting and metrics
   - Generate summary metrics for each class and overall model performance.
   - Plot precision/recall/accuracy comparisons.
5. Deployment-friendly outputs
   - Save trained weights under `model/`.
   - Store prediction examples under `outputs/` or `results/`.

## New Technology Stack
- Python 3.14
- ultralytics (YOLOv8 compatible)
- PyTorch / Torch
- OpenCV
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Migration Strategy
### Phase 1: Project preparation
- [x] Confirm current dataset structure and annotation format
- [ ] Define YOLO class labels and annotation conversion rules
- [ ] Create a modern requirements file for Python 3.14
- [ ] Prepare a `data.yaml` config for training

### Phase 2: Data conversion and verification
- [ ] Convert VOC XML labels to YOLO `.txt` label files
- [ ] Validate dataset splits: `train`, `val`, `test`
- [ ] Inspect sample images and annotations

### Phase 3: Model build and training
- [ ] Create a reproducible training pipeline using `ultralytics.YOLO`
- [ ] Train on the converted dataset
- [ ] Save the best model weights
- [ ] Evaluate on validation/test set and log metrics

### Phase 4: Inference and demo
- [ ] Build an inference script or notebook section
- [ ] Visualize detection results on sample images
- [ ] Optionally add a simple GUI/demo script using OpenCV

### Phase 5: Documentation and completion
- [ ] Update the project report/README with new workflow
- [ ] Add usage instructions and dependencies
- [ ] Confirm end-to-end execution on Python 3.14

## Compatibility Checklist
- [ ] Notebook and code run on Python 3.14
- [ ] No TensorFlow dependency remains in the modern pipeline
- [ ] YOLO training uses PyTorch-backed `ultralytics`
- [ ] Model training and inference work with the current dataset
- [ ] Output artifacts are saved in the `model/` or `outputs/` folder

## Recommended Deliverables
- `ROAD_DAMAGE_REBUILD_PLAN.md`
- `requirements-modern.txt`
- `data.yaml`
- `convert_annotations.py`
- `train_yolo.py`
- `inference_yolo.py`
- `RoadDamageModern.ipynb`
- `README.md`
- `ROAD_DAMAGE_PROGRESS_LOG.md`

## Initial Progress Log
- [x] Analyzed current notebook and report
- [x] Confirmed Python 3.14 compatibility issue with TensorFlow
- [ ] Prepared modernization plan and checklist
- [ ] Next task: create modern requirements and conversion pipeline
