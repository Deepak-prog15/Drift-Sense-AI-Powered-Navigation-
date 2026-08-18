<div align="center">
  <img src="header.svg" width="100%" alt="Drift Sense Header Animation">
</div>

<br>

> **"Drift-Sense localizes high-magnification SEM reference images inside a 10× search field to sub-pixel precision on a single CPU core. By mathematically modeling repeating semiconductor lattices, Drift-Sense recovers the sub-pixel lattice phase and intercepts catastrophic tool navigation crashes before they happen."**

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

## 📋 Submission Checklist (Applied Materials PS2)

| Required Item | Artifact Location | Description |
|---|---|---|
| **1. README.md** | `README.md` | Complete setup & generation instructions. |
| **2. Dataset Generator** | `generate.py` | Standalone generator. |
| **3. Inference Script** | `inference.py` | Primary scoring script. Outputs center coordinates. |
| **4. Core Logic** | `code/` | Clean directory without DL bloat. |
| **5. Dependencies** | `requirements.txt` | Minimal dependencies (`numpy`, `opencv-python`). |
| **6. Citations** | `CITATIONS.md` | Academic citations for physics-grounded noise models. |

## 📊 Performance Benchmarks

| Layout Category | Accuracy (≤ 5 px) | Median Error | Speed |
|---|---|---|---|
| **Solvable Layouts** | 100.0% | 0.41 px | 0.88 s |
| **Heavy Noise (2.0x)** | 100.0% | 0.68 px | 0.91 s |
| **Pure-Periodic Ambiguity**| Abstains | Lattice Phase: 0.81px | 0.82 s |
