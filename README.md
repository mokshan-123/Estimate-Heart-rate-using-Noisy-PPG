# Efficient and Robust Heart Rate Estimation from Noisy Wearable PPG Signals

This repository contains a reimplementation of the method proposed in the paper:

**"Efficient and Robust Heart Rate Estimation Approach for Noisy Wearable PPG Sensors Using Ideal Representation Learning."**

The project focuses on estimating heart rate from noisy wearable photoplethysmography (PPG) signals using a deep learning approach that learns an **idealized representation of PPG signals**. The model is trained using a Generative Adversarial Network (GAN) to transform noisy PPG signals into simplified signals that preserve heart-rate information while suppressing noise and motion artifacts.

---

# Project Overview

Photoplethysmography (PPG) signals collected from wearable sensors are often corrupted by motion artifacts, sensor noise, and physiological variability. These disturbances make accurate heart rate estimation challenging, particularly in real-world conditions.

Instead of directly estimating heart rate or denoising PPG signals, this project follows a different approach:

1. Learn an **ideal representation of PPG signals** that preserves only the information required for heart rate estimation.
2. Train a **Generative Adversarial Network (GAN)** to map noisy real PPG signals into this ideal representation.
3. Perform **systolic peak detection** on the generated signal to estimate heart rate.

This approach avoids the need for paired clean/noisy PPG data and improves robustness across different datasets, users, and noise conditions.

---

# Method Pipeline

The complete pipeline consists of several stages:

## 1. Data Collection

The method is evaluated on multiple publicly available datasets containing synchronized PPG and ECG signals.

Datasets used in the original study:

- BIDMC
- CAPNO
- WESAD
- DALIA

ECG signals are used to obtain ground truth heart rate values.

---

## 2. Signal Preprocessing

All PPG signals are standardized using the following preprocessing pipeline.

### Resampling
All signals are resampled to:

```
128 Hz
```

This ensures consistency across datasets with different sampling rates.

### Band-pass Filtering

A Butterworth band-pass filter is applied:

```
0.5 Hz – 8 Hz
```

This removes baseline drift and high-frequency noise while preserving the physiological frequency range of heartbeats.

### Window Segmentation

Signals are segmented using sliding windows:

- Window length: **4 seconds**
- Overlap: **25%**

Each window contains:

```
512 samples (128 Hz × 4 seconds)
```

### Normalization

Each segment is min-max normalized to the range:

```
[-1, 1]
```

---

# Ideal PPG Signal Generation

Instead of using real clean PPG signals as targets, the model learns from **synthetically generated ideal PPG signals**.

The ideal signal is defined as:

```
PPG_i(t) = 0.8 cos(2π fHR t + θ) − 0.35 sin(2(2π fHR t + θ))
```

Where:

- `fHR` is the heart-rate frequency
- `θ` is the phase shift

Sampling ranges:

```
fHR ∈ [0.67 Hz , 3.33 Hz]
θ ∈ [0 , 2π]
```

This corresponds approximately to:

```
40 BPM – 200 BPM
```

These ideal signals represent simplified physiological waveforms that retain the fundamental heart rate frequency but remove noise and irrelevant variability.

---

# Model Architecture

The core model is a **Generative Adversarial Network (GAN)** consisting of:

- Generator
- Discriminator

The generator converts noisy PPG signals into idealized signals.

---

# Generator Network

The generator is a **fully convolutional encoder–decoder network**.

### Encoder

| Layer | Filters | Kernel | Stride |
|------|--------|-------|-------|
| Conv1 | 64 | 16 | 2 |
| Conv2 | 128 | 16 | 2 |
| Conv3 | 256 | 16 | 2 |

Activation:
```
LeakyReLU
```

Normalization:
```
Layer Normalization
```

### Decoder

| Layer | Filters |
|------|--------|
| Deconv1 | 256 |
| Deconv2 | 128 |
| Deconv3 | 64 |

Activation:
```
ReLU
```

Output layer:

```
1 channel
tanh activation
```

Total parameters (approx):

```
2.3 Million
```

---

# Discriminator Network

The discriminator is a 1D convolutional classifier.

| Layer | Filters | Kernel | Stride |
|------|--------|-------|-------|
| Conv1 | 512 | 16 | 2 |
| Conv2 | 256 | 16 | 2 |
| Conv3 | 128 | 16 | 2 |
| Conv4 | 64 | 16 | 2 |

Activation:
```
LeakyReLU
```

Normalization:
```
Batch Normalization
```

The discriminator learns to distinguish between:

- Real ideal PPG signals
- Generator outputs

Total parameters:

```
2.7 Million
```

---

# Training Setup

Training configuration:

```
Framework: TensorFlow 2.6
Batch Size: 128
Epochs: 12
Optimizer: Adam
Initial Learning Rate: 1e-4
Learning Rate Decay: 0.9 every 10k steps
```

The GAN objective is defined as:

```
L(D,G) = E[log D(x)] + E[log(1 − D(G(z)))]
```

Where:

- `x` is an ideal PPG signal
- `z` is a noisy real PPG signal

---

# Heart Rate Estimation

After the generator produces an idealized signal, heart rate is estimated using classical signal processing.

### Peak Detection

Systolic peaks are detected using the **Elgendi peak detection algorithm**.

### Heart Rate Calculation

Heart rate is calculated from peak intervals:

```
HR = 60 / mean(RR_interval)
```

---

# Ground Truth Heart Rate

Ground truth heart rate is derived from ECG signals.

Steps:

1. Detect R-peaks using the **Hamilton algorithm**
2. Remove outliers using **Interquartile Range (IQR) filtering**
3. Compute heart rate from R–R intervals

---

# Evaluation Metric

The main evaluation metric is:

```
Mean Absolute Error (MAE)
```

Definition:

```
MAE = mean(|HR_estimated − HR_ground_truth|)
```

Measured in beats per minute (BPM).

---

# Repository Structure

```
ideal-ppg-hr/
│
├── datasets
│   ├── bidmc_loader.py
│   ├── capno_loader.py
│   ├── dalia_loader.py
│   └── wesad_loader.py
│
├── models
│   ├── generator.py
│   └── discriminator.py
│
├── utils
│   ├── preprocessing.py
│   ├── ideal_ppg.py
│   ├── peak_detection.py
│   └── metrics.py
│
├── train_gan.py
├── infer_hr.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/your-username/ideal-ppg-hr.git
cd ideal-ppg-hr
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training

To train the GAN model:

```
python train_gan.py
```

---

# Heart Rate Inference

To estimate heart rate from PPG signals:

```
python infer_hr.py
```

---

# Evaluation

To evaluate performance:

```
python evaluate.py
```

The script will report:

- Mean Absolute Error (MAE)
- Error distribution
- Predicted vs ground truth plots

---

# Key Features

- Robust heart rate estimation from noisy wearable PPG
- Ideal signal representation learning
- Lightweight model architecture suitable for mobile deployment
- Cross-dataset training for improved generalization

---

# Limitations

- Some implementation details from the original paper are not fully specified.
- Exact performance may vary depending on dataset preprocessing.
- The repository represents a research reproduction rather than the authors’ official implementation.

---

# Future Improvements

Possible extensions include:

- Multi-dataset training improvements
- Real-time inference optimization
- Mobile deployment using TensorFlow Lite
- Activity-aware heart rate estimation
- Self-supervised representation learning for PPG

---

# Reference

If you use this repository in your research, please cite the original paper:

```
Efficient and Robust Heart Rate Estimation Approach for Noisy Wearable PPG Sensors Using Ideal Representation Learning
```

---

# License

This project is intended for academic and research purposes.
