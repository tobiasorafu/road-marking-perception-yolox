# Road Marking Perception Using YOLOX

This project develops a MATLAB computer-vision pipeline for detecting road markings in dashcam images using **YOLOX** and the **CeyMo** road-marking dataset.

The detector currently focuses on two classes:

- **Straight-Left Arrow (SLA)**
- **Bus Lane (BL)**

The project covers dataset analysis, Pascal VOC annotation parsing, hard-negative selection, transfer learning, model evaluation, error analysis, data augmentation, and testing on road images outside the CeyMo dataset.

## What This Project Demonstrates

- Preparing Pascal VOC XML annotations for MATLAB object detection.
- Building training and validation datastores from the CeyMo dataset.
- Using visually similar road markings as hard negatives to reduce false detections.
- Transfer learning with a COCO-pretrained **YOLOX-tiny** detector.
- Evaluating object detection using AP, mAP, precision, recall, confusion matrices, and precision-recall curves.
- Analysing false positives and missed detections rather than relying only on headline accuracy.
- Applying brightness, contrast, and blur augmentation while preserving the meaning of direction-dependent road markings.
- Testing model generalisation on road scenes outside the training dataset.

## Detection Pipeline

```text
CeyMo images + Pascal VOC annotations
                |
                v
       Dataset preparation
                |
                v
      YOLOX-tiny transfer learning
                |
                v
        Test-set evaluation
                |
                v
          Error analysis
                |
                v
   Appearance-based augmentation
                |
                v
       Comparative evaluation
```

## Dataset Preparation

The CeyMo training annotations were parsed from Pascal VOC XML files and converted from:

```text
[xmin, ymin, xmax, ymax]
```

to MATLAB bounding-box format:

```text
[x, y, width, height]
```

All available SLA and BL positive images were used in the initial project dataset. A set of **120 non-target images** was also selected, with emphasis on visually similar arrow classes such as SA, LA, SRA and RA.

The resulting dataset contained **435 images**, split approximately 85/15 into:

| Set | Images | SLA instances | BL instances | Negative images |
| --- | ---: | ---: | ---: | ---: |
| Training | 370 | 154 | 121 | 102 |
| Validation | 65 | 26 | 21 | 18 |
| Official CeyMo test set | 788 | 61 | 49 | 678 |

The official test set was kept separate from training and validation.

## Model Configuration

The detector uses **YOLOX-tiny pretrained on COCO** and adapts the detection head to SLA and BL.

| Parameter | Value |
| --- | --- |
| Detector | YOLOX-tiny |
| Pretrained weights | tiny-coco |
| Input size | 224 x 224 x 3 |
| Classes | SLA, BL |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Mini-batch size | 70 |
| Epochs | 100 |
| Validation frequency | Every 2 iterations |

## Results

Both detectors were evaluated on the same 788-image CeyMo test set using an IoU threshold of 0.5. Detection counts below use a confidence threshold of 0.2.

| Metric | Baseline detector | Augmented detector |
| --- | ---: | ---: |
| SLA AP | 0.592 | 0.604 |
| BL AP | 0.848 | 0.821 |
| mAP | 0.720 | 0.713 |
| SLA false positives | 126 | 83 |
| BL false positives | 69 | 32 |
| Total false positives | 195 | 115 |
| SLA missed detections | 13 | 15 |
| BL missed detections | 4 | 6 |
| SLA precision | 0.276 | 0.357 |
| BL precision | 0.395 | 0.573 |

The augmented detector reduced total false positives by approximately **41%**, but it also missed slightly more road markings. The result is therefore a precision-recall trade-off rather than a universal improvement.

SLA remained the more difficult class. Similar directional arrows produced many false SLA detections, while BL was generally easier to distinguish. The 224 x 224 input resolution also reduced detail for small and distant markings.

## Augmentation Strategy

The augmented model applies random:

- brightness variation of approximately ±15%;
- contrast scaling between 0.8 and 1.2;
- mild Gaussian blur with 50% probability.

Horizontal flipping is deliberately excluded because SLA is direction-dependent. Flipping a straight-left arrow would make it resemble a straight-right arrow while leaving the original label unchanged.

## Current Development Direction

The next stage of the project focuses on improving the detector rather than simply retraining the same configuration. Planned changes include:

- increasing the quantity and diversity of training images;
- increasing input resolution to **416 x 416**;
- expanding the hard-negative set for visually similar arrows;
- selecting operating confidence thresholds using validation data;
- extending the detector to six road-marking classes: **SLA, BL, PC, JB, CL and DM**;
- evaluating performance across different road and lighting conditions;
- developing a road-video perception demonstration with inference-time measurements.

A later extension will investigate how perception outputs can be connected to supervisory logic and control-system models in MATLAB/Simulink.

## Project Contents

- `setup_project.m` - adds the source folder to the MATLAB path.
- `src/project_paths.m` - defines repository-relative dataset, model and result paths.
- `src/analyse_training_set.m` - analyses SLA, BL and overall class distribution in the CeyMo training set.
- `src/prepare_dataset.m` - parses annotations, selects positives/hard negatives and builds training/validation datastores.
- `src/train_detector.m` - trains the two-class YOLOX-tiny detector.
- `src/augment_data.m` - appearance-based augmentation function.
- `src/train_augmented_detector.m` - trains the detector with stochastic augmentation.
- `src/evaluate_detector.m` - evaluates the trained detector on the official CeyMo test set.
- `src/demo_external_images.m` - runs qualitative inference on external road images.
- `src/visualize_video_detections.m` - visualises previously generated video detection frames.

## Running the Project

1. Clone or download this repository and open it in MATLAB.
2. Run `setup_project.m`.
3. Download the CeyMo dataset from the original project repository.
4. Arrange the dataset as:

```text
data/
└── ceymo/
    ├── train/
    │   ├── images/
    │   └── bbox_annotations/
    └── test/
        ├── images/
        └── bbox_annotations/
```

5. Run `analyse_training_set.m` to inspect the training distribution.
6. Run `prepare_dataset.m` to build the training and validation datastores.
7. Run `train_detector.m` or `train_augmented_detector.m`.
8. Run `evaluate_detector.m` to evaluate the saved baseline detector.

The dataset and generated `.mat` model files are intentionally excluded from version control.

## Contributors

- Tobias Orafu
- Emediong Moffat
- Kofoworola Oyeniyi

## Dataset and Attribution

This project uses the **CeyMo: See More on Roads** dataset developed by Jayasinghe *et al.* for road-marking detection.

Original dataset repository: [oshadajay/CeyMo](https://github.com/oshadajay/CeyMo)

> O. Jayasinghe, S. Hemachandra, D. Anhettigama, S. Kariyawasam, R. Rodrigo and P. Jayasekara, "CeyMo: See More on Roads — A Novel Benchmark Dataset for Road Marking Detection," *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2022.
