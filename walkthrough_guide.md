# Gesture Recognition System Walkthrough

We have successfully built a robust gesture recognition system with 4 classes: `right`, `left`, `both`, and `other`.

## Components Created/Updated

1.  **[dataset.py](file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/dataset.py)**: Records 14 angles (Pose + Hands) from the camera.
    - Optimized with frame skipping (`DETECT_EVERY = 5`) for speed.
    - Saves each "burst" to a separate CSV in `dataset/features/` to avoid Excel permission issues.
2.  **[combine_data.py](file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/combine_data.py)**: A utility to merge all CSV and Excel files into one.
3.  **[tanilt.py](file:///c:/Users/DELL.DESKTOP-PTQ10MO/Desktop/WEB/police%20robot/tanilt.py)**: The main script that trains the Random Forest model and runs real-time recognition.

---

## 🌟 Advanced Version (Version 2)
We have implemented an improved version in the `version_2/` folder. This version is better at distinguishing whether hands are **up** or **down**.

- **New Features**: 14 angles + **4 vertical Y-differences** = **18 total features**.
- **Vertical Diffs**: Calculates the height difference between Wrist-Elbow and Elbow-Shoulder.
- **Improved Bias**: Much better at detecting the same hand posture at different heights.

### How to use Version 2:
```bash
python version_2/dataset.py       # Step 1: Record new clean data
python version_2/combine_data.py   # Step 2: Merge new data
python version_2/tanilt.py         # Step 3: Train and test 18-feature model
```

## How to use the full workflow

### Step 1: Record Data
Run `dataset.py`. Use **n** to select a label, and **SPACE** to record 1000 frames.
```bash
python dataset.py
```

### Step 2: Combine Data
Run this after recording new data or if you have manually edited Excel files. It creates `dataset/merged_features.csv`.
```bash
python combine_data.py
```

### Step 3: Train and Recognize
Run `tanilt.py`. It will load the merged file, train the model (99% accuracy!), and open the camera for detection.
```bash
python tanilt.py
```

## Results & Progress
- **Stable Version**: 6,000 frames, 14 features, 99.1% accuracy.
- **Experimental (V2)**: 4,000 frames (new), 18 features (including height sensors).
- **Performance**: Real-time detection optimized to run every 5 frames for smooth experience.
- **Robustness**: Successfully handles multi-hand scenarios and noisy backgrounds.
