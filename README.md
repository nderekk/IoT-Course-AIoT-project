# AIoT Project — Human Gesture Recognition

**University of Patras** | AIoT Course
*Derekenaris Nikolaos · Mitrogiannis Evangelos · Stauridis Goulielmos Nikolaos*

---

## 📖 Overview

This repository contains an end-to-end Artificial Intelligence of Things (AIoT) pipeline for recognizing smartphone gestures from raw 6-axis IMU (Inertial Measurement Unit) data. The data was collected using the **MetaMotionR** wrist-worn sensor kit. 

The project evaluates two distinct classification paradigms:
1. **Classical ML/Deep Learning on Raw Time-Series (`aiot_project_time_series.ipynb`)**: Using a RandomForest / 1D Convolutional Neural Network (CNN) to automatically learn spatial-temporal features from raw, windowed waveform data, compared against a flattened Random Forest baseline.
2. **Statistical Machine Learning via Feature Engineering (`feature_eng.ipynb`)**: Extracting classical Time-Domain, Frequency-Domain, and Cross-Channel features, reducing dimensionality via statistical selection, and training traditional classifiers (Random Forest, SVM, Logistic Regression).

## 🖐️ Gestures Evaluated

The dataset targets five natural smartphone navigation gestures:
1. **Scroll Up**
2. **Scroll Down**
3. **Swipe Left**
4. **Swipe Right**
5. **Texting** (Single/Two-handed natural typing)

## ⚙️ Methodology & Pipelines

### 1. Data Preprocessing
* **Outlier Cleaning**: Rolling Z-Score combined with linear interpolation to eliminate hardware spikes.
* **Scaling**: Sensor-grouped scaling (StandardScaler for Accelerometer, RobustScaler for Gyroscope) fit strictly on the training set to prevent data leakage.
* **Filtering & Segmentation**: Low-pass filtering (order=5, wn=0.4) and 300ms overlapping sliding windows (50ms overlap) to construct discrete samples.

### 2. Time-Series Pipeline (Deep Learning)
Evaluates a **1D CNN** vs. a **Random Forest** (trained on flattened windows).
* The 1D CNN preserves the 3D structure of the data `(Samples × Window_Size × Channels)` to learn local motif filters.
* Evaluated strictly using K-Fold, Leave-One-Subject-Out (LOSO) CV, and a completely unseen test dataset. 

### 3. Feature Engineering Pipeline (Statistical Learning)
* **Feature Extraction**: Extracts 76 features per window (Mean, STD, RMS, Skew, Kurtosis, ZCR, Spectral Energy, Dominant Freq, Entropy, Cross-channel Correlation, etc.).
* **Redundancy Filter**: Drops highly correlated features (Spearman correlation > 0.90/0.95).
* **Feature Selection**: Keeps the Top 15 most discriminative features using ANOVA F-statistic (`SelectKBest`).
* **Classification**: Evaluates Random Forest, SVM (RBF Kernel), and Logistic Regression. Hyperparameter tuning is performed via `RandomizedSearchCV`.

## 📊 Key Results & Insights

Models were evaluated under three protocols of increasing strictness: **5-Fold CV** (within-cohort) → **LOSO CV** (cross-subject) → **True Unseen Test Set**.

* **Random Forest (Feature Engineered)** achieved the highest within-cohort performance (~87% K-Fold F1) but suffered significantly on cross-subject generalization (~41% LOSO F1).
* **1D CNN (Raw Time-Series)** demonstrated the strongest generalization capabilities on the completely unseen test dataset (~52% Accuracy/F1), outperforming all baseline models that relied on manual feature extraction or flattened vectors.
* **Gesture Robustness**: "Texting" was overwhelmingly the most robust and transferable gesture across subjects (F1 ~0.74+ in unseen tests). In contrast, directional swipes (e.g., Swipe Left) were highly subject-dependent and collapsed during cross-subject evaluations due to variations in wrist-angle and holding posture.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* MongoDB Community Server (running locally or remotely)
* Jupyter Notebook / JupyterLab

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/IoT-Course-AIoT-project.git
cd IoT-Course-AIoT-project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database Setup & Loading the Exported Data
The raw gesture data is stored in a MongoDB database. We have provided an export of our dataset inside the `.zip` archive as `/data/aito.gestures.json`.

To replicate our environment, load this data into your own MongoDB instance using `mongoimport`:

1. Ensure your MongoDB server is running (e.g., `mongod`).
2. Run the following command from the root of the project to import the data into a database named `aiot_project` and a collection named `gestures`:
   ```bash
   mongoimport --uri="mongodb://localhost:27017" --db=aiot_project --collection=gestures --file=data/aito.gestures.json --jsonArray
   ```
   *(Note: Adjust the `--uri`, `--db`, and `--collection` parameters if you prefer different names or are using a cloud MongoDB instance).*

### 3. Configuration
1. Copy the template configuration file:
   ```bash
   cp config.yml.template config.yml
   ```
2. Open `config.yml` and ensure the MongoDB parameters match the database and collection names you used in the import step (e.g., `db: "aiot_project"`, `col: "gestures"`).

### 4. Running the Pipelines
Start your Jupyter environment:
```bash
jupyter notebook
```
* Open **`aiot_project_time_series.ipynb`** to run the Deep Learning and Flattened Random Forest pipelines.
* Open **`feature_eng.ipynb`** to run the Statistical Machine Learning pipelines on extracted features. 

*Note: Execution of the notebooks will dynamically pull the data from your local MongoDB, process the windows, and output the Confusion Matrices and Classification Reports.*

## 📞 Contact
For any questions regarding this project, please refer to the team members listed at the top of this document.