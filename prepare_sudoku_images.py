"""
Helper script to prepare sudoku.com screenshots for training data generation
Save your sudoku.com screenshots in a folder called 'sudoku_images' and run this script
"""

import cv2
import numpy as np
import os
from pathlib import Path

def extract_cells_from_sudoku(image_path, output_folder):
    """
    Extract individual cells from a sudoku image for training data
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the board
    board_contour = None
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area > (img.shape[0] * img.shape[1] * 0.1):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                board_contour = approx
                break
    
    if board_contour is None:
        print(f"Could not find board in {image_path}")
        return
    
    # Perspective transform
    rect = order_points(board_contour.reshape(4, 2))
    side = 630
    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (side, side))
    
    # Extract cells
    cell_size = side // 9
    
    filename = Path(image_path).stem
    cell_count = 0
    
    for i in range(9):
        for j in range(9):
            y1 = i * cell_size
            y2 = (i + 1) * cell_size
            x1 = j * cell_size
            x2 = (j + 1) * cell_size
            
            cell = warped[y1:y2, x1:x2]
            
            # Save cell
            cell_path = os.path.join(output_folder, f"{filename}_r{i}_c{j}.png")
            cv2.imwrite(cell_path, cell)
            cell_count += 1
    
    print(f"Extracted {cell_count} cells from {image_path}")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

if __name__ == "__main__":
    # Create folders
    input_folder = "sudoku_images"
    output_folder = "extracted_cells"
    
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input folder has images
    image_files = list(Path(input_folder).glob("*.png")) + \
                  list(Path(input_folder).glob("*.jpg")) + \
                  list(Path(input_folder).glob("*.jpeg"))
    
    if not image_files:
        print(f"\n⚠ No images found in '{input_folder}' folder!")
        print("\nHow to use this script:")
        print("1. Go to https://sudoku.com")
        print("2. Take screenshots of different sudoku puzzles")
        print(f"3. Save the screenshots in the '{input_folder}' folder")
        print("4. Run this script again")
        print("\nTip: Get 10-20 different puzzles for better training data variety!")
    else:
        print(f"Found {len(image_files)} images. Extracting cells...\n")
        
        for img_path in image_files:
            extract_cells_from_sudoku(str(img_path), output_folder)
        
        print(f"\n✓ Done! Check '{output_folder}' folder for extracted cells")
        print("\nNext steps:")
        print("1. Review the extracted cells")
        print("2. Organize them by digit (create folders: 0, 1, 2, ... 9)")
        print("3. Use them to fine-tune your model or generate more synthetic data")
