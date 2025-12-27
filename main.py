#!/usr/bin/env python3
"""
Image Sharpening Pipeline (Fixed Path + PDF Report with Summary Page)
---------------------------------------------------------------------
- Gaussian Blur (5x5, σ=1)
- Unsharp (k = 0.5, 1.0, 1.5)
- Laplacian (α = 0.2, 0.5, 0.8)
- High-Boost (A = 1.0, 1.5, 2.0, 3.0)

Outputs: PNGs + a multi-page PDF report including a single-page summary grid.
"""

import os, glob, re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Paths (robust) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "image.jpeg")  # change if needed
OUTPUT_DIR = os.path.join(BASE_DIR, "image_sharpening_outputs_v2")

GAUSS_KSIZE = (5, 5)
GAUSS_SIGMA = 1.0
UNSHARP_KS = [0.5, 1.0, 1.5]
LAPLACE_ALPHAS = [0.2, 0.5, 0.8]
HIGHBOOST_AS = [1.0, 1.5, 2.0, 3.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def read_image_gray(path: str, size_if_synth: int = 512) -> np.ndarray:
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"OpenCV failed to read image at: {path}")
        return img
    # synthetic fallback
    h = w = size_if_synth
    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    base = 0.35 + 0.55 * (0.5 * (np.sin(4*np.pi*X)*np.cos(4*np.pi*Y)) + 0.5*X)
    circle = ((X-0.3)**2 + (Y-0.4)**2) < 0.07**2
    square = (np.abs(X-0.72) < 0.08) & (np.abs(Y-0.65) < 0.08)
    imgf = base.copy()
    imgf[circle] += 0.25
    imgf[square] -= 0.2
    img = (np.clip(imgf, 0, 1)*255).astype(np.uint8)
    return img

def gaussian_blur(img, ksize=(5,5), sigma=1.0):
    return cv2.GaussianBlur(img, ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def unsharp_mask(img, k, blurred=None):
    if blurred is None:
        blurred = gaussian_blur(img, GAUSS_KSIZE, GAUSS_SIGMA)
    mask = cv2.subtract(img, blurred)
    sharp = img.astype(np.float32) + k*(img.astype(np.float32)-blurred.astype(np.float32))
    return mask, np.clip(sharp, 0, 255).astype(np.uint8)

def laplacian_sharpen(img, alpha, ksize=3):
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, ddepth=cv2.CV_32F, ksize=ksize, borderType=cv2.BORDER_REFLECT101)
    return np.clip(f - alpha*lap, 0, 255).astype(np.uint8)

def high_boost(img, A, blurred=None):
    if blurred is None:
        blurred = gaussian_blur(img, GAUSS_KSIZE, GAUSS_SIGMA)
    f = img.astype(np.float32)
    mask = f - blurred.astype(np.float32)
    out = np.clip(f + (A-1.0)*mask, 0, 255).astype(np.uint8)
    mask_vis = np.clip(mask + 128, 0, 255).astype(np.uint8)
    return mask_vis, out

def hist_plot(img, title, fname):
    fig = plt.figure(figsize=(7,4.5))
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    plt.plot(hist)
    _save_fig(fig, os.path.join(OUTPUT_DIR, fname))

def show_and_save(imgs, titles, fname):
    cols = len(imgs)
    fig = plt.figure(figsize=(6*cols,5))
    for i, (im, tl) in enumerate(zip(imgs, titles), 1):
        ax = fig.add_subplot(1, cols, i)
        ax.imshow(im, cmap="gray")
        ax.set_title(tl)
        ax.axis("off")
    _save_fig(fig, os.path.join(OUTPUT_DIR, fname))

def safe_imwrite(path, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

# ---------- PDF utilities ----------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def make_summary_grid(out_dir: str, fname: str = "summary_grid.png"):
    """Create a 3x4 grid with key outputs and save as PNG."""
    picks = [
        ("00_input.png", "Original"),
        ("01_gaussian_blur.png", "Gaussian 5x5, σ=1"),
        ("unsharp_k0.5_sharp.png", "Unsharp k=0.5"),
        ("unsharp_k1.0_sharp.png", "Unsharp k=1.0"),
        ("unsharp_k1.5_sharp.png", "Unsharp k=1.5"),
        ("laplacian_alpha0.2.png", "Laplacian α=0.2"),
        ("laplacian_alpha0.5.png", "Laplacian α=0.5"),
        ("laplacian_alpha0.8.png", "Laplacian α=0.8"),
        ("highboost_A1.0_output.png", "High-Boost A=1.0"),
        ("highboost_A1.5_output.png", "High-Boost A=1.5"),
        ("highboost_A2.0_output.png", "High-Boost A=2.0"),
        ("highboost_A3.0_output.png", "High-Boost A=3.0"),
    ]
    imgs, titles = [], []
    for fn, tl in picks:
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            imgs.append(plt.imread(p))
            titles.append(tl)

    rows, cols = 3, 4
    fig = plt.figure(figsize=(18, 12))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        if i < len(imgs):
            ax.imshow(imgs[i], cmap="gray")
            ax.set_title(titles[i], fontsize=11)
        ax.axis("off")
    fig.suptitle("Summary Grid: Key Outputs", fontsize=18)
    out_path = os.path.join(out_dir, fname)
    _save_fig(fig, out_path)
    return out_path

def make_pdf_report(out_dir: str, pdf_name: str = "Image_Sharpening_Report.pdf", title: str = "Image Sharpening Results", params: dict | None = None):
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    pdf_path = os.path.join(out_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        # cover
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.patch.set_facecolor("white")
        plt.axis("off")
        y = 0.85
        plt.text(0.5, y, title, ha="center", va="center", fontsize=24, weight="bold")
        y -= 0.12
        plt.text(0.5, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}", ha="center", va="center", fontsize=12)
        if params:
            y -= 0.08
            txt = "\n".join([f"• {k}: {v}" for k, v in params.items()])
            plt.text(0.5, y, txt, ha="center", va="top", fontsize=12, family="monospace")
        pdf.savefig(fig); plt.close(fig)

        # summary grid page
        summary_png = os.path.join(out_dir, "summary_grid.png")
        if os.path.isfile(summary_png):
            img = plt.imread(summary_png)
            fig = plt.figure(figsize=(11.7, 8.3))
            ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle("Summary Grid", fontsize=16)
            pdf.savefig(fig); plt.close(fig)

        # one image per page
        for p in sorted(pngs):
            if os.path.basename(p) == "summary_grid.png":
                continue
            img = plt.imread(p)
            fig = plt.figure(figsize=(11.7, 8.3))
            ax = fig.add_axes([0.05, 0.12, 0.9, 0.8])
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            fig.suptitle(os.path.basename(p), fontsize=14)
            pdf.savefig(fig); plt.close(fig)

    return pdf_path

def main():
    print(f"[info] Using image path: {IMAGE_PATH}")
    img = read_image_gray(IMAGE_PATH)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "00_input.png"), img)
    blur = gaussian_blur(img, GAUSS_KSIZE, GAUSS_SIGMA)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_gaussian_blur.png"), blur)
    show_and_save([img, blur], ["Original (gray)", "Gaussian (5x5, σ=1)"], "grid_input_blur.png")
    hist_plot(img, "Histogram: Original", "hist_original.png")

    for k in UNSHARP_KS:
        mask, sharp = unsharp_mask(img, k, blurred=blur)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"unsharp_k{k}_mask.png"), mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"unsharp_k{k}_sharp.png"), sharp)
        show_and_save([mask, sharp], [f"Mask (k={k})", f"Unsharp (k={k})"], f"grid_unsharp_k{k}.png")
        hist_plot(sharp, f"Histogram: Unsharp (k={k})", f"hist_unsharp_k{k}.png")

    for a in LAPLACE_ALPHAS:
        lap = laplacian_sharpen(img, a)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"laplacian_alpha{a}.png"), lap)
    show_and_save([laplacian_sharpen(img,a) for a in LAPLACE_ALPHAS],
                  [f"Laplacian α={a}" for a in LAPLACE_ALPHAS], "grid_laplacian.png")

    for A in HIGHBOOST_AS:
        mask_vis, hb = high_boost(img, A, blurred=blur)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"highboost_A{A}_mask.png"), mask_vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"highboost_A{A}_output.png"), hb)
        show_and_save([mask_vis, hb], [f"Mask (A={A})", f"High-Boost (A={A})"], f"grid_highboost_A{A}.png")

    # build summary grid
    summary_path = make_summary_grid(OUTPUT_DIR, "summary_grid.png")
    print(f"[info] Summary grid saved: {summary_path}")

    # PDF report
    pdf_path = make_pdf_report(
        OUTPUT_DIR,
        pdf_name="Image_Sharpening_Report.pdf",
        title="Unsharp, Laplacian, and High-Boost Results",
        params={
            "Image": os.path.relpath(IMAGE_PATH, start=BASE_DIR),
            "Gaussian": f"ksize={GAUSS_KSIZE}, sigma={GAUSS_SIGMA}",
            "Unsharp k": UNSHARP_KS,
            "Laplacian α": LAPLACE_ALPHAS,
            "High-boost A": HIGHBOOST_AS,
        },
    )
    print(f"[done] PDF report created at:\n  {pdf_path}")
    print(f"[done] All PNGs and PDF are in:\n  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
