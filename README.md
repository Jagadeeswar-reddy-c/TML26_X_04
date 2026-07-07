# TML 2026 Assignment 4 — Black-Box Watermark Forgery

**Team:** TML26_X_04 (CISPA)  
**Best Confirmed Score:** 0.605296 (rank 20 / 61) — `attack_v101_trustmark_hybrid.py`  
**Deadline:** 07.07.2026 23:59 CEST

---

## Quick Start — Reproduce Best Result

### 1. Install Dependencies

```bash
pip install opencv-python pillow numpy torch torchvision lpips scipy trustmark
```

### 2. Place Dataset

Download `Dataset.zip` from the course HuggingFace repo and put it in `tml2026_task4/`.  
The attack script auto-extracts it on first run.

### 3. Run the Best Attack

```bash
cd tml2026_task4
python attack_v101_trustmark_hybrid.py
```

Produces `submission_v101_trustmark_hybrid.zip` — 200 PNGs (`1.png`–`200.png`).

**Or use the pre-packaged final folder:**

```bash
cd final_submission
python submission.py        # submits submission_best.zip directly
```

### 4. Submit to Leaderboard

```bash
cd tml2026_task4
# Edit submission.py: set FILE_PATH and SUBMIT=True
python submission.py
```

---

## Method Overview

### Scoring Formula

| Symbol | Formula | Meaning |
|--------|---------|---------|
| S_det | max(0, (BitAcc − 0.5) × 2) | Detection strength |
| S_qlt | exp(−8 × LPIPS) | Visual quality |
| **S_final** | **S_det × S_qlt** | **Leaderboard score** |

### Key Discovery: WM_7 = TrustMark Q

WM_7 (targets 151–175) was identified as **TrustMark Q** (`use_ECC=False`) by running the TrustMark decoder across all 25 source images. All 25 images decoded to the **identical 100-bit message**, confirming the method.

```
WM_7 message (100 bits):
1000100100110010100110011110110010010010111110000111011111111101011001111101011011010001010011100001
```

Direct TrustMark re-encoding onto the 25 clean targets:
- **BitAcc = 100%** (decoder reads back the exact message)
- **LPIPS ≈ 0.0015** (nearly imperceptible)
- **S_final per image ≈ 0.988**

### WM_1/2/3/4/5/6/8: Iterative NLM Extraction

For the seven remaining groups, we extract a shared watermark signal from the 25 source images using **iterative Non-Local Means (NLM) denoising** and embed it into clean targets.

#### Extraction (3-signal ensemble, 8 passes)

```python
# Signal A — converging average residual
sigA = mean(src_i − NLM(src_i, h))          # pass 1
for _ in range(7):
    sigA = mean(src_i − NLM(src_i − sigA, h))  # passes 2–8

# Signal B — per-image refined residual
sigB[i] = src_i − NLM(src_i − sigA, h=4)

# Signal C — Gaussian high-pass on source average
sigC = avg(src_i) − GaussianBlur(avg(src_i), σ=1.0)

# Ensemble
ens = 1.0 × norm(sigA) + 0.5 × norm(sigB[i]) + 0.5 × norm(sigC)
```

#### Embedding

```python
# Bisect for optimal scale: LPIPS(target, target + scale×ens) ≤ LPIPS_budget
scale = bisect_search(...)
forged = clip(target + scale × ens, 0, 255)
```

### Per-Method Configuration

| Method | Targets | NLM H | LPIPS Budget | BitAcc (est.) | S_final (est.) |
|--------|---------|-------|--------------|---------------|----------------|
| WM_1   | 1–25    | 2     | 0.020        | ~81%          | ~0.550         |
| WM_2   | 26–50   | 8     | 0.025        | ~81%          | ~0.550         |
| WM_3   | 51–75   | 2     | 0.010        | ~81%          | ~0.550         |
| WM_4   | 76–100  | 2     | 0.010        | ~81%          | ~0.550         |
| WM_5   | 101–125 | 2     | 0.005        | ~81%          | ~0.550         |
| WM_6   | 126–150 | 2     | 0.010        | ~81%          | ~0.550         |
| **WM_7** | **151–175** | — | **~0.0015** | **100%** | **~0.988** |
| WM_8   | 176–200 | 2     | 0.025        | ~81%          | ~0.550         |

---

## Score History

| Version | Technique | Score | Notes |
|---------|-----------|-------|-------|
| Baseline | Alpha blend (50/50) | ~0.010 | Terrible LPIPS |
| v28 | 4-pass NLM + sigB + sigC | 0.461 | First NLM approach |
| v34 | 8-pass NLM + sigB + sigC | 0.467 | More passes help |
| **v54** | **8-pass NLM, halved LPIPS** | **0.502** | LPIPS insight — scale-invariance |
| v89 | 2× LPIPS (test) | ≤0.502 | Confirmed amplitude invariance |
| **v101** | **TrustMark WM_7 + NLM others** | **0.605** | **Method identification (+0.103)** |

### Key Insights

1. **NLM scale-invariance** (v54 breakthrough): The watermark decoder does not depend on embedding amplitude — only the pattern matters. Minimizing LPIPS improves S_qlt without hurting S_det. Going from LPIPS=0.041 → 0.016 gave +7.5% score.

2. **Method identification** (v101 breakthrough): WM_7 is TrustMark Q. Direct re-encoding with the extracted 100-bit message gives 100% BitAcc at LPIPS≈0.0015. This alone contributed +0.103 to the final score.

3. **Iterative refinement** (8 passes): Each NLM pass with the current estimate removes content noise while preserving the coherent watermark signal. Passes 1→8 progressively increase the signal-to-noise ratio.

---

## File Structure

```
TML26_X_04/
├── README.md                          ← this file
├── CLAUDE.md                          ← AI assistant guidance
├── Assignment_4_-_Watermark_Forging.pdf
├── final_submission/                  ← READY-TO-USE FOLDER
│   ├── attack_best.py                 ← best attack code (v101)
│   ├── submission.py                  ← ready-to-submit script
│   └── submission_best.zip            ← pre-built 200-image zip (score 0.605296)
└── tml2026_task4/
    ├── Dataset.zip                    ← place here before running
    ├── attack_v54_lowlpips.py         ← v54 (0.502 baseline)
    ├── attack_v101_trustmark_hybrid.py← BEST CONFIRMED (0.605296)
    ├── attack_v102_tm_halflpips.py    ← TrustMark + 0.5× LPIPS
    ├── attack_v103_tm_quarterlpips.py ← TrustMark + 0.25× LPIPS
    ├── attack_v104_tm_minlpips.py     ← TrustMark + 0.1× LPIPS
    ├── submission.py                  ← leaderboard submission
    └── submission_v101_trustmark_hybrid.zip ← best confirmed zip
```

---

## References

- Yang et al., "Can Simple Averaging Defeat Modern Watermarks?" (NeurIPS 2024)
- Dong et al., "WMCopier: Forging Invisible Image Watermarks on Arbitrary Images" (NeurIPS 2025)
- Souček et al., "Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models" (NeurIPS 2025)
- Bui et al., "TrustMark: Universal Watermarking for Arbitrary Resolution Images" (AAAI 2024)
