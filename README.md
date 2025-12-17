# Induction Motor Fault Diagnosis using Machine Learning

An industry-oriented project that combines **Electrical Engineering (Induction Machines)** with **Machine Learning and Data Analytics** to build an intelligent fault diagnosis and condition monitoring system.

This project demonstrates how **core electrical machine knowledge** and **modern ML techniques** can be integrated to solve real-world industrial maintenance problems.

---

## 🔧 Why This Project?

Induction motors are widely used in industries such as manufacturing, power plants, and automation.  
Failures in motor bearings and internal components can cause unexpected downtime and high maintenance costs.

Traditional maintenance approaches are reactive or schedule-based.  
This project applies **data-driven machine learning** to detect motor faults early using **vibration signal analysis**.

---

## ⚙️ What This Project Does

- Uses **raw vibration signals from induction motors**
- Applies **signal processing techniques** rooted in electrical machines theory
- Extracts meaningful **time-domain and frequency-domain features**
- Trains and evaluates **machine learning models** for fault classification
- Identifies motor health conditions automatically

---

## ⚡ Electrical Engineering (EEE) Focus

- Induction motor condition monitoring
- Bearing fault analysis (Inner race, Outer race, Ball defect)
- Vibration-based fault detection
- Frequency-domain analysis of motor signals
- Practical predictive maintenance concepts

This reflects **real practices used by electrical and maintenance engineers** in industry.

---

## 🧠 IT / Machine Learning Focus

- Time-series data preprocessing
- Feature engineering using FFT and statistical analysis
- Supervised machine learning models:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Cross-validation and model comparison
- Model persistence and evaluation

This demonstrates **applied ML skills**, not just theoretical implementation.

---

## 🧪 Dataset

- **Case Western Reserve University (CWRU) Induction Motor Bearing Dataset**
- Raw vibration sensor data
- Industry-standard benchmark dataset

---

## ▶️ How to Run

```bash
python train_fault_diagnosis.py --mat-dir path/to/data/raw --out-dir models
