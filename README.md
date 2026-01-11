# 🫀 Heart Attack Risk Detection from ECG Images (CNN–LSTM)

> **Academic / Undergraduate Project**  
> **Purpose:** Proof-of-concept exploration of deep learning for cardiac risk estimation from ECG images  
> **Not intended for clinical or diagnostic use**

---

## 📌 Project Overview

This project explores the feasibility of using deep learning to **estimate heart-attack (myocardial infarction) risk from ECG images**. A hybrid **CNN–LSTM** architecture is used to learn:

- **Spatial ECG morphology** (via Convolutional Neural Networks)
- **Sequential cardiac patterns** (via Long Short-Term Memory networks)

The model outputs a **risk probability**, indicating whether an ECG image is more likely to belong to a **higher-risk (“risky”)** or **lower-risk (“normal”)** category.

This work was completed as an **undergraduate engineering project**, with emphasis on:
- End-to-end ML pipeline design
- Representation learning from medical signals
- Correct framing of limitations and safe use

---

## 🎯 Problem Statement

Early identification of high-risk cardiac patterns in ECGs can support triage and further clinical evaluation. While many approaches operate directly on raw ECG waveforms, this project investigates an alternative:

> **Can ECG images alone contain enough information to enable coarse risk stratification using deep learning?**

This scenario reflects practical settings where:
- Only printed or rendered ECG images are available
- Raw waveform data is inaccessible
- Rapid screening is required

---

## 🧠 Model Architecture

The system uses a **CNN–LSTM hybrid model**:

### 1. Convolutional Neural Network (CNN)
- Extracts spatial features from ECG images
- Learns waveform morphology such as:
  - ST-segment deviations
  - QRS complex shapes
  - Relative peak positions

### 2. LSTM (Bidirectional)
- Treats CNN feature maps as a temporal sequence
- Captures sequential dependencies across the ECG signal

### 3. Output Layer
- Single sigmoid-activated logit
- Produces a **risk probability** between 0 and 1

> This architecture was chosen to reflect both **local ECG structure** and **temporal cardiac dynamics**.

---

## 📂 Dataset Structure

The project assumes a **folder-based dataset layout**:


ECG_ROOT/

normal/

ecg_001.png

ecg_002.png

…

risky/

ecg_101.png

ecg_102.png

…

### Label Semantics

| Label | Meaning |
|------|--------|
| `0` (normal) | ECGs without clear signs of acute cardiac abnormality |
| `1` (risky)  | ECGs exhibiting abnormal patterns associated with elevated cardiac risk |

⚠️ **Important:**  
Labels are derived from the dataset used for academic purposes and **do not represent medical diagnoses**.

---

## ⚙️ Configuration (What to Modify)

Key configuration values are defined at the top of the training script:

```python
ECG_ROOT        # Path to ECG image dataset
CLASS_TO_IDX   # Folder → label mapping
IMG_SIZE       # Image resolution (default: 224)
BATCH_SIZE     # Training batch size
LEARNING_RATE  # Optimizer learning rate
NUM_EPOCHS     # Number of training epochs

```



## 📊 Evaluation Approach

Given the academic nature of the project, evaluation focuses on **feasibility**, not clinical performance.

Reported metrics may include:
- Validation accuracy
- ROC-AUC (when class balance permits)
- Qualitative behavior of predictions

No claims of clinical validity or diagnostic accuracy are made.

---

## ⚠️ Limitations

This project has several important limitations:

- Limited dataset size and diversity
- Image-based ECG representation instead of raw waveforms
- No patient-level data splitting
- No external or clinical validation
- Not calibrated for real-world decision thresholds

As such, the model **must not be used for medical decision-making**.

---

## 🔬 Ethical Considerations

- This project is strictly for **educational and research exploration**
- Outputs represent **risk likelihood**, not diagnoses
- Misuse in clinical settings could result in harm
- Future medical AI systems must comply with regulatory and ethical standards

---

## 🔮 Future Improvements

If revisiting this work today, key improvements would include:
- Training directly on raw ECG waveforms (1D CNNs or transformers)
- Patient-level data splits to avoid data leakage
- Calibration and uncertainty estimation
- Explainability aligned with clinical interpretation
- External dataset validation

---

## 📚 Related Research

This project aligns with existing literature on:
- CNN and CNN–LSTM models for ECG-based myocardial infarction detection
- Image-based ECG representations for deep learning
- Hybrid spatial–temporal neural architectures in medical AI

---

## 🧑‍🎓 Author & Context

- **Author:** Preetam Jena  
- **Context:** Undergraduate engineering capstone project  
- **Focus:** Machine learning, signal processing, and healthcare AI fundamentals

