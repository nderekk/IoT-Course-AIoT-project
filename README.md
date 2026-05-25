# AIoT Project 2 — Human Gesture Recognition

**University of Patras** | AIoT Course | May 2026

*Derekenaris Nikolaos · Mitrogiannis Evangelos · Stauridis Goulielmos Nikolaos*

---

## 📖 Overview

An end-to-end Artificial Intelligence of Things (AIoT) pipeline for recognising five smartphone gestures from raw 6-axis IMU data, collected via the **MetaMotionR** wrist-worn sensor kit (Bosch BMI160, 100 Hz).

Two classification paradigms are implemented and compared:

1. **Time-Series Pipeline** (`aiot_project_time_series.ipynb`) — a 1D CNN and a flattened Random Forest trained directly on raw windowed waveforms, preserving or collapsing the temporal dimension respectively.
2. **Feature Engineering Pipeline** (`feature_eng.ipynb`) — 76 time- and frequency-domain features extracted per window, reduced via correlation filtering and ANOVA selection, then fed to RF, SVM, and Logistic Regression classifiers.

All models are evaluated under three protocols of increasing strictness: **5-Fold CV → LOSO CV → True Unseen Test Set**, to cleanly separate within-cohort performance from real cross-subject generalisation.

---

## 🖐️ Gesture Classes

| Gesture | Description |
|---|---|
| Scroll Up | Finger drags downward on screen to move content up |
| Scroll Down | Finger drags upward on screen to move content down |
| Swipe Left | Horizontal glide from right to left |
| Swipe Right | Horizontal glide from left to right |
| Texting | Natural typing on the smartphone virtual keyboard |

---

## ⚙️ Methodology

### Data Collection
- **Device:** MetaMotionR (BMI160 IMU), 6-axis (accelerometer + gyroscope), 100 Hz
- **Protocol:** 3 subjects × 5 gestures × 1-minute isolated recording clips
- **Total:** 75 minutes of training data + a separate held-out test session collected on a later date
- **Conditions:** Indoor, with natural variation in posture and phone-holding angle across subjects

### Preprocessing
- **Outlier cleaning:** Rolling Z-score (window=50, σ=3.0) with linear interpolation for hardware spike removal
- **Scaling:** StandardScaler for accelerometer axes, RobustScaler for gyroscope axes — both fit on training data only
- **Segmentation:** Sliding window of 200 samples (2 s) with 50-sample hop
- **Filtering:** 5th-order zero-phase Butterworth low-pass filter (Wn=0.4, 20 Hz cutoff)

### Time-Series Pipeline
- **RF (flattened):** Raw windows flattened to 1200-dimensional vectors (200 samples × 6 axes) and fed to a Random Forest
- **1D CNN:** Two Conv1D blocks with BatchNorm, ReLU, MaxPool, and Dropout, followed by a fully connected head; operates on 3D input `(Batch × Channels × Length)` in PyTorch

### Feature Engineering Pipeline
- **Extraction:** 76 features per window — mean, std, RMS, min, max, skewness, kurtosis, zero-crossing rate, spectral energy per axis and sensor, plus SMA and Vector Magnitude
- **Redundancy filter:** Spearman correlation — features with r > 0.90 dropped
- **Selection:** Top 15 features retained via ANOVA F-statistic (`SelectKBest`)
- **Tuning:** `RandomizedSearchCV` (50 iterations, 5-fold CV) following Bergstra & Bengio (2012)

---

## 📊 Results

### Time-Series Pipeline

| Model | K-Fold Acc | K-Fold F1 | LOSO Acc | LOSO F1 | Unseen Acc | Unseen F1 |
|---|---|---|---|---|---|---|
| RF (Flattened) | 77% | 0.77 | 28% | 0.29 | 23% | 0.14 |
| 1D CNN | 91% | 0.91 | 43% | 0.44 | 52% | 0.48 |

### Feature Engineering Pipeline

| Model | K-Fold Acc | K-Fold F1 | LOSO Acc | LOSO F1 | Unseen Acc | Unseen F1 |
|---|---|---|---|---|---|---|
| Random Forest | 86% | 0.86 | 41% | 0.41 | 40% | 0.40 |
| SVM (RBF) | 64% | 0.64 | 31% | 0.32 | — | — |
| Logistic Regression | 70% | 0.71 | 39% | 0.41 | — | — |

### Key Takeaways
- **1D CNN** achieves the best unseen-test performance (52%), demonstrating that learned temporal filters transfer partially across subjects
- **Feature Engineering RF** generalises better than the flattened RF (40% vs 23%), confirming compact statistical features are less susceptible to subject-specific signal morphology than raw concatenated windows
- **Texting** is the most robust class across all models and protocols (F1 up to 1.00 in K-Fold), owing to its distinctive high-frequency micro-movement fingerprint
- **Swipe Left** is the most subject-dependent class, collapsing to F1 ≈ 0.00–0.09 in LOSO and unseen tests across all models
- The ~45-point drop from K-Fold to LOSO/unseen is consistent across all models, pointing to **subject-dependent overfitting** as the primary limiting factor — a predictable consequence of a 2-subject training cohort under uncontrolled recording conditions

---

## 📁 Project Structure

```
├── aiot_project_time_series.ipynb  # Time-series pipeline (RF + 1D CNN)
├── feature_eng.ipynb               # Feature engineering pipeline (RF, SVM, LR)
├── eda.ipynb                       # Exploratory data analysis
├── utils.py                        # Core signal processing utilities
├── utils_visual.py                 # Plotting and visualisation utilities
├── config.yml.template             # MongoDB connection template
├── requirements.txt                # Python dependencies
└── data/
    └── aiot.gestures.json          # MongoDB collection export (mongoimport-ready)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- MongoDB Community Server (local or remote)
- Jupyter Notebook / JupyterLab

### 1. Clone & Install
```bash
git clone https://github.com/nderekk/IoT-Course-AIoT-project.git
cd IoT-Course-AIoT-project
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Load the Dataset
Ensure MongoDB is running, then import the provided collection export:
```bash
mongoimport --uri="mongodb://localhost:27017" \
            --db=aiot_project \
            --collection=gestures \
            --file=data/aiot.gestures.json \
            --jsonArray
```

### 3. Configure
```bash
cp config.yml.template config.yml
```
Open `config.yml` and set `db` and `col` to match the names used above (`aiot_project` and `gestures`).

### 4. Run
```bash
jupyter notebook
```
- `aiot_project_time_series.ipynb` — time-series pipeline
- `feature_eng.ipynb` — feature engineering pipeline
- `eda.ipynb` — exploratory analysis

---

## 📞 Contact

For questions, contact any of the team members listed above.
