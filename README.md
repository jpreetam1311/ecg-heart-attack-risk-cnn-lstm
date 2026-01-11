# 🫀 Cardiac Risk Signal Detection from ECG Images (CNN–LSTM)

> **Academic Project (Undergraduate Capstone)**  
> **Purpose:** Proof-of-concept exploration of AI-driven signal extraction and risk stratification from visual time-series data  
> **Not intended for clinical, diagnostic, or medical use**

---

## 📌 Project Overview

This project explores the feasibility of using deep learning to **extract risk-related signals from ECG images** and produce a **coarse-grained risk score**. A hybrid **CNN–LSTM** architecture is used to learn:

- **Spatial patterns** from rendered signal images (via Convolutional Neural Networks)
- **Sequential dependencies** across the signal timeline (via Long Short-Term Memory networks)

The model outputs a **probabilistic risk score**, indicating whether an input belongs to a **higher-risk** or **lower-risk** category.

While the domain context is ECG data, the project is intentionally framed as a **generalizable AI pipeline for signal detection and prioritization**, applicable to any scenario involving:
- Image-based time-series representations
- Large volumes of signals requiring triage
- Human-in-the-loop review and decision support

---

## 🎯 Problem Framing (Generalized)

Many real-world systems rely on **rendered or visual representations of signals** rather than raw sensor data. Examples include:
- Exported charts or screenshots
- Legacy systems without waveform access
- Monitoring tools that surface images rather than structured data

This project explores the question:

> **Can visual signal representations alone contain enough information to enable coarse risk stratification and prioritization using deep learning?**

The goal is not precise prediction, but **early filtering and prioritization**, reducing the burden on downstream human or automated workflows.

---

## 🧠 System Architecture

The system uses a **CNN–LSTM hybrid architecture**, chosen to balance interpretability, feasibility, and extensibility.

### 1. Convolutional Neural Network (CNN)
- Extracts spatial features from signal images
- Learns local morphology such as:
  - Shape deviations
  - Relative peak positions
  - Structural irregularities

### 2. Sequence Modeling Layer (Bidirectional LSTM)
- Treats CNN feature maps as an ordered sequence
- Captures temporal dependencies across the signal timeline

### 3. Output Layer
- Sigmoid-activated output
- Produces a **risk likelihood score** in the range [0, 1]

This architecture reflects a common **AI design pattern** for combining local feature extraction with temporal context.

---

## 🧩 Product-Oriented Use Cases

Although demonstrated on ECG images, this approach generalizes to multiple product and platform scenarios:

- Automated triage of high-risk or anomalous signals
- Prioritization of events for analyst or operator review
- Reducing review volume while preserving signal quality
- Feeding downstream dashboards, alerts, or workflow automation
- Supporting decision-making under time or data constraints

The model is best viewed as a **decision-support component**, not a fully autonomous system.

---

## 📂 Dataset Structure

The project assumes a simple **folder-based dataset layout**:



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
- Outputs represent **relative risk likelihood**, not diagnoses
- Not suitable for safety-critical or regulated deployment
- Responsible AI systems must include governance, monitoring, and human oversight

---

## 🔮 Future Improvements

If revisiting this work today, key improvements would include:
- Training directly on raw ECG waveforms (1D CNNs or transformers)
- Patient-level data splits to avoid data leakage
- Uncertainty estimation and confidence calibration
- Explainability aligned with domain interpretation
- Evaluation on larger, more diverse datasets

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



