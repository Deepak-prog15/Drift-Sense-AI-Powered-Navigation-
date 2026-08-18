<div align="center">
  <img src="header.svg" width="100%" alt="Drift Sense Header Animation">
</div>

<br>

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-OpenCV%20|%20NumPy-orange?style=for-the-badge)
![Compute](https://img.shields.io/badge/Compute-Single%20CPU%20Core-success?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25%20Solvable-brightgreen?style=for-the-badge)
![Latency](https://img.shields.io/badge/Latency-<1.0s-blueviolet?style=for-the-badge)
![Target](https://img.shields.io/badge/Target-Wafer%20Inspection-red?style=for-the-badge)

---

# 📌 Project Overview

This project implements an **AI-Powered Navigation-Error Recovery System** for Scanning Electron Microscope (SEM) wafer inspection tools. 

Due to stage drift and beam hysteresis, high-magnification SEM captures often miss their intended targets. **Drift-Sense** acts as a robust recovery layer, localizing a small $1000 \times 1000$ high-magnification reference image inside a $10 \times$ low-magnification search field. 

The design goal is to strictly balance:
* ⚡ Ultra-low latency (CPU only)
* 🧠 Robustness to extreme SEM Poisson noise
* 🎯 Sub-pixel accuracy without Deep Learning hallucinations
* 🏭 Safe abstention on pure-periodic ambiguous arrays

---

# 🎯 Objectives

* Relocalize lost wafer inspection targets accurately.
* Survive high physical degradation (Line-Edge Roughness, Charging).
* Eliminate heavy GPU dependencies by utilizing advanced classical CV.
* Output exact sub-pixel $(x, y)$ coordinates.
* Prevent catastrophic tool crashes through calibrated abstention.

---

# 🖼️ Drift-Sense Pipeline

```text
Low-Mag Search Image
        │
        ▼
Bicubic Downsampling & Blur Matching
        │
        ▼
Gradient-Domain Projection (Sobel)
        │
        ▼
Normalized Cross-Correlation (Grad-NCC)
        │
        ▼
Phase-Correlation Sub-pixel Refinement
        │
        ▼
Target (x, y) Prediction
```

---

## 🚀 1-Minute Reviewer Quickstart

Verify all submission items with zero configuration.

<details open>
<summary><b>Setup & Verification Instructions</b></summary>
<br>

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate Synthetic Test Dataset**
Generates 30 realistic image pairs with recorded ground truth:
```bash
python generate.py --style DRAM --num 30 --out data/dram_eval
```

**3. Run Critical Inference Script**
Outputs exact single `(x, y)` coordinate:
```bash
python inference.py --ref data/dram_eval/images/00000_ref.png --search data/dram_eval/images/00000_search.png
# Expected Output format: 851.89, 815.91
```

</details>

---

## 📋 Submission Checklist (Applied Materials PS2)

| Required Item | Artifact Location | Description |
|---|---|---|
| **1. README.md** | `README.md` | Complete setup & generation instructions. |
| **2. Dataset Generator** | `generate.py` | Standalone generator. |
| **3. Inference Script** | `inference.py` | Primary scoring script. Outputs center coordinates. |
| **4. Core Logic** | `code/` | Clean directory without DL bloat. |
| **5. Dependencies** | `requirements.txt` | Minimal dependencies (`numpy`, `opencv-python`). |
| **6. Citations** | `CITATIONS.md` | Academic citations for physics-grounded noise models. |

---

## 📊 Performance Benchmarks

| Layout Category | Accuracy (≤ 5 px) | Median Error | Speed |
|---|---|---|---|
| **Solvable Layouts** | 100.0% | 0.41 px | 0.88 s |
| **Heavy Noise (2.0x)** | 100.0% | 0.68 px | 0.91 s |
| **Pure-Periodic Ambiguity**| Abstains | Lattice Phase: 0.81px | 0.82 s |
