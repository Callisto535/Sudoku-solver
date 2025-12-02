# Sudoku Solver (Python, OpenCV, CNN)

A computer-vision powered Sudoku solver that detects the puzzle from an image, recognizes digits with a convolutional neural network (CNN), and solves the grid using a classic backtracking algorithm.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Workflow](#project-workflow)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Demo](#demo)
8. [Folder Structure](#folder-structure)
9. [Methods Used](#methods-used)
10. [Dataset & Model](#dataset--model)
11. [Contributors](#contributors)
12. [License](#license)

---

## Introduction

This project is a **Python-based Sudoku Solver** that takes an image of a 9×9 Sudoku puzzle, automatically detects the grid, recognizes the digits, and computes the solution.

The pipeline combines:
- **Image processing** (OpenCV) to find and extract the Sudoku board and individual cells.
- **Deep learning** (CNN model) to recognize digits under various lighting and background conditions.
- **Backtracking search** to solve the Sudoku logically and efficiently.

The goal is to provide an end-to-end example of computer vision + machine learning + classic algorithms working together in a real-world task.

---

## Features

- ✅ Upload a Sudoku image and automatically detect the 9×9 grid.
- ✅ Robust grid detection using contours, perspective transforms, and line detection.
- ✅ Digit recognition using a trained CNN model (with synthetic and augmented data).
- ✅ Handles different backgrounds, grid styles, and mild noise.
- ✅ Solves the puzzle using a backtracking algorithm.
- ✅ Returns the solved grid as JSON (and/or overlay-ready format).

---

## Tech Stack

- **Language:** Python 3.x
- **Computer Vision:** OpenCV
- **Deep Learning:** TensorFlow / Keras (CNN for digit classification)
- **Numerical Computing:** NumPy
- **Web / API (if used):** Flask

---

## Project Workflow

High-level step-by-step workflow:

1. **Image Input**
   - User provides an image of a 9×9 Sudoku puzzle (e.g., screenshot or photo).

2. **Pre-processing** (OpenCV)
   - Convert to grayscale.
   - Apply contrast enhancement (e.g., CLAHE).
   - Apply blurring and thresholding to highlight the grid and digits.

3. **Grid Detection**
   - Detect contours and identify the largest Sudoku-like square.
   - Apply perspective transform to obtain a top-down view of the board.

4. **Cell Extraction**
   - Divide the board into a 9×9 grid.
   - Extract each cell image.

5. **Digit Recognition (CNN)**
   - Pre-process cell images (resize/normalize).
   - Run each cell through the trained CNN model.
   - Build the initial Sudoku board matrix (0 for empty cells).

6. **Solving (Backtracking)**
   - Use a classic backtracking algorithm to fill the board.
   - Validate each step against Sudoku constraints (row, column, subgrid).

7. **Output**
   - Return the solved grid (e.g., as JSON).
   - Optionally, overlay the solution back onto the original image.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Callisto535/Sudoku-solver.git
cd Sudoku-solver
```

2. **Create and activate a virtual environment (recommended)**

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

If you have a `requirements.txt` file, install with:

```bash
pip install -r requirements.txt
```

Otherwise, make sure at least the following are installed:

```bash
pip install opencv-python numpy tensorflow flask
```

---

## Usage

### 1. Run the Web App (if using Flask)

```bash
python app.py
```

- The app will typically run at `http://127.0.0.1:5000`.
- Open the URL in your browser, upload a Sudoku image, and submit.

### 2. Use the Solver Logic Directly (Example)

If you want to call the solver logic from a script, the flow is generally:

```python
from app import process_sudoku_image  # example entrypoint

board, solved_board = process_sudoku_image("path/to/sudoku.png")
print("Detected board:")
print(board)
print("Solved board:")
print(solved_board)
```

(Adjust the import and function names based on the actual implementation in `app.py`.)

---

## Demo

You can add example images and screenshots here.

- **Input Sudoku Image**  
  `![Sudoku Input](demo/sudoku_input.png)`

- **Detected Grid & Digits**  
  `![Detected Grid](demo/detected_grid.png)`

- **Solved Sudoku Overlay**  
  `![Solved Sudoku](demo/solved_sudoku.png)`

*(Replace the paths above with real images once you add them to the repository, e.g. in a `demo/` folder.)*

---

## Folder Structure

A simplified view of the project structure:

```text
Sudoku-solver/
├─ app.py                  # Main application (image processing + API / UI)
├─ train_model.py          # Script to train the CNN digit recognizer
├─ digit_model.h5          # Trained CNN model weights
├─ sudoku_images/          # Original Sudoku screenshots / images
│  ├─ easy.png
│  ├─ medium.png
│  ├─ hard.png
│  ├─ expert.png
│  ├─ extreme.png
│  └─ master.png
├─ extracted_cells/        # Individual 28x28 cell images extracted from boards
├─ FINAL_VERSION_SUMMARY.txt
├─ HOW_TO_COLLECT_IMAGES.txt
└─ README.md
```

---

## Methods Used

### 1. Image Processing (OpenCV)

- **Grayscale conversion** and **contrast enhancement** (e.g., CLAHE).
- **Noise reduction** using blurring/denoising filters.
- **Adaptive thresholding** and **Otsu thresholding** to segment digits and grid.
- **Contour detection** to find the outer Sudoku border.
- **Perspective transform** to warp the board to a top-down, square view.
- **Grid splitting** into 9×9 cells.

### 2. Machine Learning (CNN Digit Classifier)

- CNN architecture built with TensorFlow/Keras.
- Input size typically 28×28 grayscale images.
- Softmax output over 10 classes (digits 0–9 or 1–9 depending on setup).
- Trained on a combination of standard digit datasets (e.g. MNIST) and synthetic Sudoku-like digits.

### 3. Solving Algorithm (Backtracking)

- Classic **backtracking search** to fill in empty cells.
- At each empty cell, try digits 1–9 and check:
  - Row constraint (no duplicates in the row),
  - Column constraint,
  - 3×3 subgrid constraint.
- Backtrack on conflicts until a full valid solution is found.

---

## Dataset & Model

### Synthetic Digit Dataset

To make the model robust to real-world Sudoku images (e.g., different backgrounds, grid lines, and fonts), the training data includes **synthetic digits** generated with:

- Varying **background colors** (white, light gray, light blue, etc.).
- Varying **digit colors** (black, dark gray, dark blue).
- Multiple **fonts and sizes**.
- Random **rotations**, **shifts**, and **scaling**.
- Light **noise**, **blur**, and simulated **grid lines** near the cell borders.

This synthetic data helps the CNN generalize beyond clean datasets like MNIST to real Sudoku screenshots.

### Data Augmentation

During training, standard image augmentation is applied, such as:

- Random rotations within a small angle range.
- Zoom-in / zoom-out.
- Width and height shifts.
- Shear transformations.

This augmentation greatly improves the robustness of the model to slight misalignments and distortions in the input cells.

### Trained Model

- Saved as `digit_model.h5` in the project root.
- Can be loaded directly in `app.py` to run predictions on extracted cell images.

---

## Contributors

- **Callisto535** – Project author and maintainer

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or issue on GitHub.

---

## License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2025 Callisto535

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
