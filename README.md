# Image Enhancement & Sharpening Pipeline 📸

A classical Computer Vision–based image enhancement system exploring multiple sharpening techniques to improve perceptual quality, edge fidelity, and visual clarity. This project implements and analyzes Gaussian Smoothing, Unsharp Masking, Laplacian Sharpening, and High-Boost Filtering, serving as a strong practical foundation for modern deep learning–based restoration and GAN-powered enhancement systems.

---

## 🚀 Features

* **Grayscale Processing:** Supports grayscale image reading with a fallback synthetic image generator.
* **Multiple Enhancement Techniques:**
    * **Gaussian Blur:** Noise smoothing baseline.
    * **Unsharp Masking:** Edge enhancement via difference mapping.
    * **Laplacian Sharpening:** High-frequency detail amplification.
    * **High-Boost Filtering:** Aggressive edge strengthening with adjustable factors.
* **Parameterized Experimentation:** Easily adjust intensity effects to observe sharpening trade-offs.
* **Modular Architecture:** Robust and organized pipeline designed for scalability.
* **Visual Analysis:** Automatically generates comparison grids and organized directory outputs.

---

## 🧠 Tech & Concepts Used

### Computer Vision Fundamentals
* **Spatial Filtering:** Utilizing kernels for pixel-wise transformations.
* **Edge Enhancement:** Identifying and amplifying high-frequency components.
* **Contrast Preservation:** Balancing sharpness without blowing out pixel intensities.

### Image Processing Concepts
* **Gaussian Smoothing:** Used to reduce high-frequency noise before sharpening.
* **Laplacian Edge Amplification:** Using second-order derivatives to detect rapid intensity changes.
* **Mask Generation:** Creating a difference map (High-Pass) for boosting.

### Tools & Libraries
* **Python:** Core programming language.
* **OpenCV:** Primary image processing library.
* **NumPy:** Efficient matrix and array manipulations.
* **Matplotlib:** Generating visual comparison grids.

---

## 📂 Project Structure

```text
├── main.py                      # Main enhancement pipeline
├── image.jpeg                   # Input image (user provided)
├── image_sharpening_outputs_v2  # Generated outputs
│   ├── Gaussian Output          # Smoothed baseline images
│   ├── Unsharp Outputs          # Subtraction-based sharpening
│   ├── Laplacian Outputs        # Derivative-based sharpening
│   ├── High-Boost Outputs       # Aggressive amplification
│   └── Comparison Grids         # Side-by-side analysis results

```

## ⚙️ How It Works

### 🔹 Gaussian Blur
Creates a smoothed baseline image. By removing high-frequency noise first, we ensure that the sharpening process doesn't amplify unwanted artifacts or grain.

### 🔹 Unsharp Masking
Enhances edges by calculating a "mask" (Original - Blurred) and adding it back to the original image.



### 🔹 Laplacian Sharpening
Highlights regions of rapid intensity change. It uses a second-order derivative kernel to identify fine details and add them back to the source.



### 🔹 High-Boost Filtering
A generalization of unsharp masking. It uses an amplification factor to strengthen the original image's contribution, allowing for much more aggressive sharpening than standard masking.

---

## 🧪 Results & Observations
This pipeline enables experimentation with:

* **Sharpening Strengths:** Finding the "sweet spot" before ringing artifacts appear.
* **Edge vs. Artifact Trade-off:** Observing how noise reacts to different kernels.
* **Sensitivity & Stability:** Analyzing which methods are most robust against low-resolution or pixelated inputs.


---

## ▶️ Running the Project

---

### 1. Install Dependencies
```bash
pip install opencv-python numpy matplotlib
```
### 2. Add Input Image
Place an image.jpeg in the root directory.

(Note: If no image is found, the script will automatically generate a synthetic sample for testing.)

### 3. Run the script
```bash
    python main.py
```

