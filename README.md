# TML 2026 — Assignment 4: Watermark Forgery Attack
## Reproducing the Best Leaderboard Result

**Course:** Trustworthy Machine Learning 2026, CISPA  
**Supervisors:** Adam Dziedzic and Franziska Boenisch  
**TAs:** Maitri Shah and Nima Dindarsafa  
**Team ID:** TML26_X_04  
**Best Public Score:** 0.605296 (rank 20 / 61)

---

## Folder Contents

```
final_submission/
├── README.md            ← this file
├── attack_best.py       ← attack script that produced the best result
├── submission.py        ← leaderboard submission script (pre-configured)
└── submission_best.zip  ← pre-built zip: 200 forged PNGs (score 0.605296)
```

> **To submit immediately without re-running:** `python submission.py` (SUBMIT is already True).

---

## Step-by-Step Reproduction

### Step 1 — Install Dependencies

```bash
pip install opencv-python pillow numpy torch torchvision lpips scipy trustmark
```

Tested on Python 3.12 with:
- `trustmark==0.9.1`
- `lpips==0.1.4`
- `opencv-python==4.x`
- `torch>=2.0`

> **Note:** Installing `trustmark` may downgrade numpy from 2.x to 1.26.4. This is expected and does not break lpips or opencv.

---

### Step 2 — Place the Dataset

Download `Dataset.zip` from the course HuggingFace page and place it at:

```
tml2026_task4/Dataset.zip
```

The script auto-extracts it on first run into `clean_targets/` and `watermarked_sources/`.

---

### Step 3 — Run the Attack

**Run from inside this `final_submission/` folder:**

```bash
python attack_best.py
```

The script automatically finds the dataset at `../tml2026_task4/` and writes the output **directly into this folder** as `submission_best.zip` (overwriting the pre-built copy).

> **Skip this step entirely** if you just want to submit — `submission_best.zip` is already included and pre-verified (score 0.605296).

Runtime: approximately 25–35 minutes on CPU.

---

### Step 4 — Submit to the Leaderboard

```bash
cd ../final_submission
python submission.py
```

`submission.py` is already configured:
- `FILE_PATH = Path("submission_best.zip")` (pre-built output)
- `API_KEY = "568be0178160b1148f06f177d7d56b9a"`
- `SUBMIT = True`

Cooldown: 60 minutes between accepted submissions; 2 minutes after a failed one.

---

## Target Mapping (Fixed by Assignment)

| Source Group | Forge onto |
|-------------|-----------|
| WM_1 | `1.png` – `25.png` |
| WM_2 | `26.png` – `50.png` |
| WM_3 | `51.png` – `75.png` |
| WM_4 | `76.png` – `100.png` |
| WM_5 | `101.png` – `125.png` |
| WM_6 | `126.png` – `150.png` |
| WM_7 | `151.png` – `175.png` |
| WM_8 | `176.png` – `200.png` |

---

## Approach

The attack uses **two strategies** depending on whether the watermarking method was identified.

### Strategy 1 — Direct Re-encoding (WM_7 only)

WM_7 was identified as **TrustMark Q** (`use_ECC=False`) by running the TrustMark decoder on all 25 source images. All 25 decoded to the **identical 100-bit message**, confirming the method:

```
1000100100110010100110011110110010010010111110000111011111111101011001111101011011010001010011100001
```

We re-encode this message directly onto each clean target:

```python
from trustmark import TrustMark
tm = TrustMark(verbose=False, use_ECC=False)
watermarked = tm.encode(clean_image, WM7_MESSAGE, MODE='binary')
```

Result: **BitAcc = 100%, LPIPS ≈ 0.0015, S_final ≈ 0.988 per image.**

`MODE='binary'` is critical — without it, TrustMark ASCII-encodes the string to 704 bits, which mismatches the model's 100-bit capacity.

### Strategy 2 — Iterative NLM Extraction (WM_1/2/3/4/5/6/8)

For the seven unidentified groups, we extract the shared watermark signal using **8-pass iterative Non-Local Means (NLM) denoising**, then embed it into clean targets:

**Signal A (8-pass iterative average):**
```python
sigA = mean(src_i − NLM(src_i, h))              # pass 1
for _ in range(7):
    sigA = mean(src_i − NLM(src_i − sigA, h))   # passes 2–8
```

**Signal B (per-image refined residual):**
```python
sigB[i] = src_i − NLM(src_i − sigA, h=4)
```

**Signal C (Gaussian high-pass on source average):**
```python
sigC = mean(src) − GaussianBlur(mean(src), σ=1.0)
```

**Ensemble and embedding:**
```python
ensemble = 1.0 × norm(sigA) + 0.5 × norm(sigB[i]) + 0.5 × norm(sigC)
scale = bisect_search(LPIPS(target, target + scale×ensemble) ≤ LPIPS_budget)
forged = clip(target + scale × ensemble, 0, 255)
```

### Per-Group Hyperparameters

| Group | NLM h (sigA) | NLM h (sigB) | LPIPS Budget | Rationale |
|-------|-------------|-------------|--------------|-----------|
| WM_1 | 2 | 4 | 0.020 | Moderate coherence |
| WM_2 | 8 | 4 | 0.025 | Low coherence; H=8 extracts coarser pattern |
| WM_3 | 2 | 4 | 0.010 | High coherence (cons ≈ 1.0) |
| WM_4 | 2 | 4 | 0.010 | High coherence (cons ≈ 1.0) |
| WM_5 | 2 | 4 | 0.005 | Very high coherence; zero high-freq energy |
| WM_6 | 2 | 4 | 0.010 | High coherence |
| WM_7 | — | — | ~0.0015 | **TrustMark Q direct encoding** |
| WM_8 | 2 | 4 | 0.025 | Low coherence |

**LPIPS bisect parameters:** `MAX_SCALE=5.0`, `BISECT_ITERS=8`, `REF_STD=5.0`

**Ensemble weights:** `W_NLM_AVG=1.0`, `W_NLM_IND=0.5`, `W_GAUSS=0.5`

---

## Scoring

| Metric | Formula | Our Result |
|--------|---------|------------|
| Detection Strength | S_det = max(0, (BitAcc − 0.5) × 2) | WM_7: 1.0 / others: ~0.62 |
| Visual Quality | S_qlt = exp(−8 × LPIPS) | WM_7: 0.988 / others: ~0.887 |
| **Final Score** | **S_final = S_det × S_qlt** | **0.605296 (public leaderboard)** |

The public leaderboard uses 30% of samples; the private leaderboard uses the remaining 70%.

---

## Key Findings

1. **Method identification beats extraction:** Directly re-encoding the known TrustMark message gave 100% BitAcc for WM_7 vs ~81% from NLM extraction — a +20-point improvement on that group alone, adding +0.103 to the overall score.

2. **Decoders are amplitude-invariant:** Reducing LPIPS from 0.041 → 0.016 improved S_qlt with no drop in BitAcc (confirmed by ablation). The pattern shape matters; the embedding amplitude does not (above a minimum threshold).

3. **8-pass NLM converges:** Each iterative pass refines the watermark estimate by subtracting the current estimate before denoising. By pass 8, content noise has largely cancelled, leaving a clean watermark signal.

---

## References

- Yang et al., "Can Simple Averaging Defeat Modern Watermarks?" NeurIPS 2024
- Dong et al., "WMCopier: Forging Invisible Image Watermarks on Arbitrary Images" NeurIPS 2025
- Souček et al., "Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models" NeurIPS 2025
- Bui et al., "TrustMark: Universal Watermarking for Arbitrary Resolution Images" AAAI 2024
- Craver et al., "Resolving Rightful Ownerships with Invisible Watermarking Techniques" IEEE JSAC 1998
- Kutter & Voloshynovskiy, "The Watermark Copy Attack" SPIE 2000
