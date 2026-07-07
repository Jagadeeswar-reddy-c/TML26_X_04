"""
Best attack — TrustMark direct encoding for WM_7 + Iterative NLM for WM_1/2/3/4/5/6/8.

Leaderboard score: 0.605296 (rank 20/61, Team TML26_X_04)

Run from this directory (final_submission/):
    python attack_best.py

Requires Dataset.zip to be placed at ../tml2026_task4/Dataset.zip
Outputs: submission_best.zip  (200 PNGs, 1.png–200.png)

KEY DISCOVERY — WM_7 is TrustMark Q (use_ECC=False):
  All 25 WM_7 source images decode to the same 100-bit message with TrustMark.
  Direct re-encoding gives BitAcc=100%, LPIPS≈0.0015, S_final≈0.988 per image.

WM_1/2/3/4/5/6/8 — Iterative NLM extraction (8 passes):
  Treats the watermark as consistent signal across 25 sources; content noise cancels.
"""

import zipfile
from pathlib import Path

import cv2
import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from trustmark import TrustMark

# ── Paths ─────────────────────────────────────────────────────────────────────
# Dataset lives in ../tml2026_task4/ relative to this script's location
SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / ".." / "tml2026_task4"
CLEAN_DIR    = DATA_DIR / "clean_targets"
SOURCE_DIR   = DATA_DIR / "watermarked_sources"
DATASET_ZIP  = DATA_DIR / "Dataset.zip"

TEMP_OUT_DIR = SCRIPT_DIR / "submission_temp_best"
FILE_PATH    = SCRIPT_DIR / "submission_best.zip"

# ── WM_7 message (100 bits, verified by majority vote across 25 sources) ──────
WM7_MESSAGE = "1000100100110010100110011110110010010010111110000111011111111101011001111101011011010001010011100001"

# ── NLM hyperparameters ────────────────────────────────────────────────────────
N_ITER = 8
SIGA_H = {
    "WM_1": 2, "WM_2": 8, "WM_3": 2, "WM_4": 2,
    "WM_5": 2, "WM_6": 2, "WM_8": 2,
}
SIGB_H = 4
METHOD_LPIPS = {
    "WM_1": 0.020, "WM_2": 0.025, "WM_3": 0.010, "WM_4": 0.010,
    "WM_5": 0.005, "WM_6": 0.010, "WM_8": 0.025,
}
W_NLM_AVG = 1.0; W_NLM_IND = 0.5; W_GAUSS = 0.5
NLM_TSIZE = 7; NLM_SSIZE = 21
REF_STD = 5.0; BISECT_ITERS = 8; MAX_SCALE = 5.0

NLM_CATEGORIES = [
    ("WM_1",   1,  25), ("WM_2",  26,  50), ("WM_3",  51,  75), ("WM_4",  76, 100),
    ("WM_5", 101, 125), ("WM_6", 126, 150), ("WM_8", 176, 200),
]

# ── Setup ──────────────────────────────────────────────────────────────────────
if not CLEAN_DIR.exists():
    print(f"Extracting {DATASET_ZIP} ...")
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(DATA_DIR)
TEMP_OUT_DIR.mkdir(exist_ok=True)

# ── LPIPS metric ───────────────────────────────────────────────────────────────
lpips_fn = lpips_lib.LPIPS(net="alex")
def local_lpips(a, b):
    def t(x): return torch.from_numpy(x.astype(np.float32)/127.5-1).permute(2,0,1).unsqueeze(0)
    with torch.no_grad(): return float(lpips_fn(t(a), t(b)))

# ── NLM helpers ────────────────────────────────────────────────────────────────
def nlm_denoised(arr, h):
    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(u8, None, h, h, NLM_TSIZE, NLM_SSIZE).astype(np.float32)

def norm(sig):
    s = sig.std(); return sig * (REF_STD / s) if s > 1e-6 else sig

def resize_sig(s, tw, th):
    if s.shape[:2] == (th, tw): return s
    v = np.clip(s + 128, 0, 255).astype(np.uint8)
    return np.array(Image.fromarray(v).resize((tw, th), Image.LANCZOS)).astype(np.float32) - 128.0

def find_scale(target_arr, ensemble, target_lp):
    tgt_u8 = target_arr.astype(np.uint8)
    hi_lp = local_lpips(tgt_u8, np.clip(target_arr + MAX_SCALE * ensemble, 0, 255).astype(np.uint8))
    if hi_lp < target_lp: return MAX_SCALE
    lo, hi = 0.0, MAX_SCALE
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2
        lp = local_lpips(tgt_u8, np.clip(target_arr + mid * ensemble, 0, 255).astype(np.uint8))
        if lp < target_lp: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def iterative_extract(srcs, h_siga, h_sigb=4, n_iter=N_ITER):
    denoised_p1 = [nlm_denoised(src, h_siga) for src in srcs]
    residuals_p1 = [src - d for src, d in zip(srcs, denoised_p1)]
    sig_a = np.mean(residuals_p1, axis=0)
    ind_std = np.mean([r.std() for r in residuals_p1])
    print(f"    pass1 cons (H={h_siga}): {sig_a.std()/ind_std:.4f}", flush=True)
    for it in range(1, n_iter):
        improved = [src - nlm_denoised(src - sig_a, h_siga) for src in srcs]
        sig_a = np.mean(improved, axis=0)
        print(f"    pass{it+1} cons (H={h_siga}): {sig_a.std()/ind_std:.4f}", flush=True)
    improved_sigb = [src - nlm_denoised(src - sig_a, h_sigb) for src in srcs]
    return sig_a, improved_sigb

# ── Run ────────────────────────────────────────────────────────────────────────
total_processed = 0
all_final_lpips = []

# Part 1: WM_7 — TrustMark Q direct encoding
print("\nWM_7 → TrustMark Q direct encoding (BitAcc=100% verified)")
tm = TrustMark(verbose=False, use_ECC=False)
wm7_lpips = []
for i in range(151, 176):
    tgt = Image.open(CLEAN_DIR / f"{i}.png").convert("RGB")
    watermarked = tm.encode(tgt, WM7_MESSAGE, MODE='binary')
    lp = local_lpips(np.array(tgt), np.array(watermarked))
    wm7_lpips.append(lp); all_final_lpips.append(lp)
    watermarked.save(TEMP_OUT_DIR / f"{i}.png")
    total_processed += 1
print(f"  mean LPIPS={np.mean(wm7_lpips):.4f}  S_qlt≈{np.exp(-8*np.mean(wm7_lpips)):.3f}", flush=True)

# Part 2: WM_1/2/3/4/5/6/8 — Iterative NLM
for source_wm, target_start, target_stop in NLM_CATEGORIES:
    h_siga = SIGA_H[source_wm]
    target_lp = METHOD_LPIPS[source_wm]
    print(f"\n{source_wm} → NLM H={h_siga} (8-pass), LPIPS budget={target_lp}", flush=True)

    src_paths = sorted((SOURCE_DIR / source_wm).glob("*.png"))
    srcs = [np.array(Image.open(p).convert("RGB")).astype(np.float32) for p in src_paths]
    avg_wm = np.mean(srcs, axis=0)

    sig_A, improved_sigb_list = iterative_extract(srcs, h_siga, SIGB_H, N_ITER)
    sig_C = avg_wm - np.stack([gaussian_filter(avg_wm[:,:,c], 1.0) for c in range(3)], -1)

    method_lpips = []
    for i in range(target_start, target_stop + 1):
        tgt = np.array(Image.open(CLEAN_DIR / f"{i}.png").convert("RGB")).astype(np.float32)
        th, tw = tgt.shape[:2]
        sA = resize_sig(sig_A, tw, th)
        sB = resize_sig(improved_sigb_list[i - target_start], tw, th)
        sC = resize_sig(sig_C, tw, th)
        ens = W_NLM_AVG*norm(sA) + W_NLM_IND*norm(sB) + W_GAUSS*norm(sC)
        scale = find_scale(tgt, ens, target_lp)
        forged = np.clip(tgt + ens * scale, 0, 255).astype(np.uint8)
        lp = local_lpips(tgt.astype(np.uint8), forged)
        method_lpips.append(lp); all_final_lpips.append(lp)
        Image.fromarray(forged).save(TEMP_OUT_DIR / f"{i}.png")
        total_processed += 1

    ml = np.mean(method_lpips)
    print(f"  mean LPIPS={ml:.4f}  S_qlt≈{np.exp(-8*ml):.3f}")

# Pack zip
print(f"\nForged {total_processed} images.")
ml = np.mean(all_final_lpips)
print(f"Overall mean LPIPS: {ml:.4f}  →  S_qlt ≈ {np.exp(-8*ml):.3f}")
with zipfile.ZipFile(FILE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(img_path, arcname=img_path.name)
print(f"Done → {FILE_PATH}")
