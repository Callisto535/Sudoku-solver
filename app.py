import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template_string
import io

app = Flask(__name__)

# ===========================
# 0. Frontend Template
# ===========================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sudoku Solver</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; text-align: center; padding: 20px; }
        h1 { color: #333; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .upload-section { margin-bottom: 30px; padding: 20px; border: 2px dashed #ccc; border-radius: 5px; }
        button { background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .results { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 30px; display: none; }
        .board-container { text-align: center; }
        table { border-collapse: collapse; margin: 0 auto; }
        td { width: 30px; height: 30px; border: 1px solid #ccc; text-align: center; font-size: 18px; }
        td:nth-child(3n) { border-right: 2px solid black; }
        td:nth-child(1) { border-left: 2px solid black; }
        tr:nth-child(3n) td { border-bottom: 2px solid black; }
        tr:nth-child(1) td { border-top: 2px solid black; }
        tr:nth-child(9) td { border-bottom: 2px solid black; }
        td:nth-child(9) { border-right: 2px solid black; }
    </style>
</head>
<body>
<div class="container">
    <h1>📸 Sudoku Solver</h1>
    <div class="upload-section">
        <input type="file" id="fileInput" accept="image/*"><br><br>
        <button onclick="uploadAndSolve()" id="solveBtn">Solve Puzzle</button>
        <p style="color:red" id="errorMessage"></p>
    </div>
    <div class="results" id="resultsArea">
        <div class="board-container"><h3>Detected</h3><table id="originalGrid"></table></div>
        <div class="board-container"><h3>Solved</h3><table id="solvedGrid"></table></div>
    </div>
</div>
<script>
    async function uploadAndSolve() {
        const fileInput = document.getElementById('fileInput');
        const errorMsg = document.getElementById('errorMessage');
        const resultsArea = document.getElementById('resultsArea');
        const solveBtn = document.getElementById('solveBtn');
        if (!fileInput.files[0]) { errorMsg.innerText = "Select a file first."; return; }
        errorMsg.innerText = ""; solveBtn.disabled = true; solveBtn.innerText = "Processing...";
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/solve', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Error");
            renderBoard(data.original, 'originalGrid', []);
            renderBoard(data.solved, 'solvedGrid', data.original);
            resultsArea.style.display = 'flex';
        } catch (err) { errorMsg.innerText = err.message; } 
        finally { solveBtn.disabled = false; solveBtn.innerText = "Solve Puzzle"; }
    }
    function renderBoard(board, id, original) {
        const table = document.getElementById(id); table.innerHTML = "";
        for (let i = 0; i < 9; i++) {
            const tr = document.createElement('tr');
            for (let j = 0; j < 9; j++) {
                const td = document.createElement('td');
                if (board[i][j] !== 0) {
                    td.innerText = board[i][j];
                    if (original.length > 0 && original[i][j] === 0) {
                        td.style.color = "#007bff"; td.style.fontWeight = "bold";
                    }
                }
                tr.appendChild(td);
            }
            table.appendChild(tr);
        }
    }
</script>
</body>
</html>
"""

try:
    model = load_model('digit_model.h5')
    print("Model loaded successfully.")
except:
    print("WARNING: 'digit_model.h5' not found. Please run train_model.py first!")

# ===========================
# 1. Computer Vision Pipeline
# ===========================

def pre_process_image(img, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Multiple blur and threshold attempts
    blur1 = cv2.GaussianBlur(gray, (5, 5), 1)
    blur2 = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Try multiple threshold methods
    thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    _, thresh3 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine thresholds
    thresh = cv2.bitwise_or(thresh1, thresh2)
    thresh = cv2.bitwise_or(thresh, thresh3)
    
    # Morphological operations to close gaps in grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def find_board_contours(thresh, img_shape):
    """Enhanced board detection with multiple strategies"""
    h, w = img_shape[:2]
    img_area = h * w
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Strategy 1: Look for largest quadrilateral
    for cnt in contours[:15]:  # Check top 15 contours
        area = cv2.contourArea(cnt)
        
        # Area should be at least 10% of image but not more than 95%
        if area < img_area * 0.10 or area > img_area * 0.95:
            continue
            
        peri = cv2.arcLength(cnt, True)
        
        # Try multiple epsilon values for approximation
        for epsilon_factor in [0.02, 0.015, 0.025, 0.03, 0.01]:
            approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
            
            if len(approx) == 4:
                # Verify it's roughly square
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = float(cw) / ch if ch > 0 else 0
                
                # Should be roughly square (between 0.7 and 1.4 ratio)
                if 0.7 <= aspect_ratio <= 1.4:
                    return approx
    
    # Strategy 2: Look for grid pattern using line detection
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100, minLineLength=min(h,w)//4, maxLineGap=20)
    
    if lines is not None and len(lines) > 20:
        # Found enough lines, likely a grid exists
        # Find bounding box of all line endpoints
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_points.extend([(x1, y1), (x2, y2)])
        
        if all_points:
            all_points = np.array(all_points)
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)
            
            # Add margin
            margin = 10
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # Return as quadrilateral
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
    
    # Strategy 3: Return largest contour if it's big enough
    if contours:
        largest = contours[0]
        if cv2.contourArea(largest) > img_area * 0.15:
            x, y, cw, ch = cv2.boundingRect(largest)
            return np.array([[x, y], [x+cw, y], [x+cw, y+ch], [x, y+ch]], dtype=np.int32)
    
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(img, location):
    rect = order_points(location.reshape(4, 2))
    # Increased resolution to 630 (70px per cell) for even better detail
    side = 630  # Increased from 540 to 630
    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    # Use better interpolation for clearer digits
    return cv2.warpPerspective(img, M, (side, side), flags=cv2.INTER_CUBIC)

def detect_grid_lines(warped):
    """Detect actual grid lines to find precise cell boundaries"""
    try:
        # Preprocess for line detection
        blur = cv2.GaussianBlur(warped, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h, w = thresh.shape
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//9, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//9))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line positions
        h_positions = []
        v_positions = []
        
        # Detect horizontal lines
        h_projection = np.sum(horizontal_lines, axis=1)
        for i in range(len(h_projection)):
            if h_projection[i] > w * 0.3 * 255:  # Strong horizontal line
                if not h_positions or i - h_positions[-1] > h//20:
                    h_positions.append(i)
        
        # Detect vertical lines
        v_projection = np.sum(vertical_lines, axis=0)
        for i in range(len(v_projection)):
            if v_projection[i] > h * 0.3 * 255:  # Strong vertical line
                if not v_positions or i - v_positions[-1] > w//20:
                    v_positions.append(i)
        
        # If we found roughly 10 lines each (9 cells + borders), use them
        if 8 <= len(h_positions) <= 12 and 8 <= len(v_positions) <= 12:
            return sorted(h_positions), sorted(v_positions)
        
    except Exception as e:
        print(f"Grid line detection failed: {e}")
    
    return None, None

def extract_digit(cell):
    try:
        # Ensure cell is uint8 type
        if cell.dtype != np.uint8:
            cell = cell.astype(np.uint8)
        
        # Check if cell is empty or too small
        if cell.size == 0 or cell.shape[0] < 10 or cell.shape[1] < 10:
            return None
        
        # Store original for analysis
        original = cell.copy()
        
        # Analyze cell characteristics
        cell_mean = np.mean(cell)
        cell_std = np.std(cell)
        cell_min = np.min(cell)
        cell_max = np.max(cell)
        
        # 1. Smart preprocessing based on cell brightness
        # If very bright (white/light background), enhance contrast more aggressively
        if cell_mean > 180:  # Light background
            # Aggressive contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3,3))
            cell = clahe.apply(cell)
            cell = cv2.normalize(cell, None, 0, 255, cv2.NORM_MINMAX)
        else:
            # Normal contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            cell = clahe.apply(cell)
            cell = cv2.normalize(cell, None, 0, 255, cv2.NORM_MINMAX)
            
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
    
    # 2. Advanced Denoising
    # First pass: Non-local means
    cell = cv2.fastNlMeansDenoising(cell, None, 10, 7, 21)
    
    # Second pass: Bilateral filter for edge preservation
    cell = cv2.bilateralFilter(cell, 5, 50, 50)
    
    # 3. Enhanced Sharpening with unsharp mask
    gaussian = cv2.GaussianBlur(cell, (0, 0), 2.0)
    cell = cv2.addWeighted(cell, 1.5, gaussian, -0.5, 0)
    cell = np.clip(cell, 0, 255).astype(np.uint8)
    
    # 4. ENSEMBLE THRESHOLDING - multiple methods with voting
    thresh_results = []
    weights = []
    
    # Method 1: Adaptive Gaussian (most reliable)
    for block_size in [11, 13, 15]:
        for C in [2, 3, 4]:
            try:
                t = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, block_size, C)
                thresh_results.append(t)
                weights.append(1.2)  # Higher weight for Gaussian
            except:
                pass
    
    # Method 2: Adaptive Mean
    for block_size in [11, 15]:
        for C in [3, 4]:
            try:
                t = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY_INV, block_size, C)
                thresh_results.append(t)
                weights.append(0.8)
            except:
                pass
    
    # Method 3: Otsu (global optimal threshold)
    _, t_otsu = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_results.append(t_otsu)
    weights.append(1.0)
    
    # Method 4: Niblack-style local threshold
    if cell_std > 10:  # Only if there's enough variation
        dynamic_thresh = max(0, min(255, int(cell_mean - cell_std * 0.7)))
        _, t_dynamic = cv2.threshold(cell, dynamic_thresh, 255, cv2.THRESH_BINARY_INV)
        thresh_results.append(t_dynamic)
        weights.append(0.9)
    
    # Method 5: Percentile-based threshold
    percentile_val = np.percentile(cell, 40)
    _, t_percentile = cv2.threshold(cell, percentile_val, 255, cv2.THRESH_BINARY_INV)
    thresh_results.append(t_percentile)
    weights.append(0.7)
    
    # WEIGHTED VOTING SYSTEM
    if len(thresh_results) > 0:
        thresh_sum = np.zeros_like(thresh_results[0], dtype=np.float32)
        total_weight = 0
        
        for t, w in zip(thresh_results, weights):
            thresh_sum += (t.astype(np.float32) / 255.0) * w
            total_weight += w
        
        # Normalize and apply adaptive threshold
        thresh_avg = thresh_sum / total_weight
        
        # Keep pixels that appear in >40% of weighted votes
        thresh = (thresh_avg >= 0.4).astype(np.uint8) * 255
    else:
        return None
    
    # 5. Intelligent Inversion Check
    if np.sum(thresh) > (thresh.size * 255 * 0.5):
        thresh = cv2.bitwise_not(thresh)
    
    # 6. Safe Border Detachment - clear margins to detach from grid lines
    h, w = thresh.shape
    margin = int(min(h, w) * 0.1)  # 10% margin
    thresh[0:margin, :] = 0
    thresh[h-margin:h, :] = 0
    thresh[:, 0:margin] = 0
    thresh[:, w-margin:w] = 0
    
    # 7. Advanced Morphological Operations
    # Remove very small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    
    # Fill small holes within digits
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Gradient to enhance edges (helps with thin digits)
    kernel_gradient = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel_gradient)
    
    # Combine original with gradient
    thresh = cv2.bitwise_or(thresh, gradient)

    # 8. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    # 9. ULTRA-SMART SCORING LOGIC with multiple criteria
    center_x, center_y = w // 2, h // 2
    candidates = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # A. Basic Filters
        if area < 20: continue  # Very lenient minimum area
        if cw < 2 or ch < 6: continue  # Very lenient minimum dimensions
        
        # B. Centrality Score
        cx, cy = x + cw // 2, y + ch // 2
        dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        centrality = 1.0 - (dist_from_center / max_dist)
        
        # C. Aspect Ratio Check
        ratio = cw / float(ch)
        if ratio > 3.0 or ratio < 0.1: continue  # Ignore obvious lines
        
        # Aspect ratio score (digits are usually taller than wide, or squarish)
        if 0.3 <= ratio <= 1.2:  # Good digit proportions
            ratio_score = 1.0
        elif ratio < 0.3:  # Very thin (like '1')
            ratio_score = 0.8
        else:  # Wider
            ratio_score = 0.6
        
        # D. Solidity (compactness)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # E. Size relative to cell
        size_ratio = area / (h * w)
        if size_ratio < 0.02 or size_ratio > 0.75: continue
        
        # Size score (prefer medium-sized blobs)
        if 0.08 <= size_ratio <= 0.35:
            size_score = 1.0
        elif 0.05 <= size_ratio < 0.08 or 0.35 < size_ratio <= 0.5:
            size_score = 0.8
        else:
            size_score = 0.6
        
        # F. Extent (how well it fills the bounding box)
        extent = area / float(cw * ch) if (cw * ch) > 0 else 0
        
        # G. Combined Multi-Factor Score
        score = (
            (area ** 0.7) *           # Area importance
            (centrality ** 2.0) *     # Centrality is critical
            (solidity ** 0.4) *       # Compactness
            (ratio_score ** 0.5) *    # Aspect ratio
            (size_score ** 0.6) *     # Size appropriateness
            (extent ** 0.3)           # Bounding box filling
        )
        
        # Must be reasonably centered
        if centrality > 0.2:
            candidates.append({
                'contour': cnt,
                'score': score,
                'centrality': centrality,
                'area': area,
                'solidity': solidity
            })
    
    if not candidates: return None
    
    # Select best candidate
    best = max(candidates, key=lambda x: x['score'])
    best_cnt = best['contour']

    # 10. Extract and Process the Best Candidate
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    
    # Smart dilation based on digit size
    x, y, cw, ch = cv2.boundingRect(best_cnt)
    area_ratio = cv2.contourArea(best_cnt) / (cw * ch) if (cw * ch) > 0 else 0
    
    if area_ratio < 0.5:  # Thin/sparse digit - dilate more
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
    else:  # Normal/thick digit - dilate less
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    # 11. INTELLIGENT RESIZE to 28x28
    digit = np.zeros((28, 28), dtype=np.uint8)
    try:
        roi = mask[y:y+ch, x:x+cw]
        
        # Check if ROI has enough content
        pixel_count = cv2.countNonZero(roi)
        if pixel_count < 10: return None
        
        # Calculate optimal target size (preserve aspect ratio better)
        max_dim = max(cw, ch)
        target_size = 20
        
        if max_dim < 12:  # Very tiny digit
            target_size = 24  # Make it bigger
        elif max_dim < 20:  # Small digit
            target_size = 22
        elif max_dim < 35:  # Medium digit
            target_size = 20
        else:  # Large digit
            target_size = 18
        
        scale = target_size / max_dim
        nh = max(1, min(27, int(ch * scale)))
        nw = max(1, min(27, int(cw * scale)))
        
        # High-quality interpolation
        if max_dim < 15:  # Upscaling
            roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_CUBIC)
        elif max_dim > 45:  # Significant downscaling
            # Two-pass resize for better quality
            intermediate_size = (int(nw * 1.5), int(nh * 1.5))
            roi = cv2.resize(roi, intermediate_size, interpolation=cv2.INTER_AREA)
            roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
        else:  # Normal size
            roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        if roi.size == 0: return None
        
        # Perfect centering with padding
        pad_x = (28 - nw) // 2
        pad_y = (28 - nh) // 2
        digit[pad_y:pad_y+nh, pad_x:pad_x+nw] = roi
        
        # Final polish
        # Ensure digit is thick enough
        if pixel_count < 100:  # Thin digit
            digit = cv2.dilate(digit, np.ones((2,2), np.uint8), iterations=1)
        
        # Apply final binary threshold
        _, digit = cv2.threshold(digit, 100, 255, cv2.THRESH_BINARY)
        
        # Anti-aliasing blur for smoother edges (helps recognition)
        digit = cv2.GaussianBlur(digit, (3,3), 0.5)
        
    except Exception as e:
        print(f"Error in digit extraction: {e}")
        return None

    return digit.astype("float32").reshape(1, 28, 28, 1) / 255.0

# ===========================
# 2. Logic & Solver
# ===========================

def sanitize_board(board, confidences):
    """Remove conflicting numbers, prioritizing the one with lower confidence."""
    changes = True
    while changes:
        changes = False
        units = []
        for r in range(9): units.append([(r,c) for c in range(9)])
        for c in range(9): units.append([(r,c) for r in range(9)])
        for br in range(3):
            for bc in range(3):
                units.append([(br*3+i, bc*3+j) for i in range(3) for j in range(3)])
        
        for unit in units:
            seen = {}
            for r, c in unit:
                val = board[r][c]
                if val == 0: continue
                if val in seen:
                    pr, pc = seen[val]
                    if confidences[r][c] > confidences[pr][pc]:
                        board[pr][pc] = 0; confidences[pr][pc] = 0
                    else:
                        board[r][c] = 0; confidences[r][c] = 0
                    changes = True
                else: seen[val] = (r, c)
    return board

def solve_sudoku(board):
    find = find_empty(board)
    if not find: return True
    row, col = find
    for i in range(1, 10):
        if valid(board, i, (row, col)):
            board[row][col] = i
            if solve_sudoku(board): return True
            board[row][col] = 0
    return False

def valid(board, num, pos):
    # Check row & col
    for i in range(9):
        if board[pos[0]][i] == num and pos[1] != i: return False
        if board[i][pos[1]] == num and pos[0] != i: return False
    # Check box
    box_x, box_y = pos[1] // 3, pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if board[i][j] == num and (i, j) != pos: return False
    return True

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0: return (i, j)
    return None

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/solve', methods=['POST'])
def process_sudoku():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        in_memory = io.BytesIO(); file.save(in_memory)
        data = np.frombuffer(in_memory.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if image is None: return jsonify({"error": "Invalid image"}), 400
        
        # Auto-rotate if image is too wide or tall
        h, w = image.shape[:2]
        if w > h * 1.5:  # Very wide image, rotate
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            print("Rotated image 90° clockwise")
        elif h > w * 1.5:  # Very tall image might need rotation
            # Check if rotating helps
            pass

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing approaches
        processed = pre_process_image(image)
        contour = find_board_contours(processed, image.shape)
        
        # If first attempt fails, try with different preprocessing
        if contour is None:
            print("First detection failed, trying alternative preprocessing...")
            # Try with simpler preprocessing
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, simple_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contour = find_board_contours(simple_thresh, image.shape)
        
        # Last resort: use entire image
        if contour is None:
            print("Board detection failed, using entire image...")
            h, w = image.shape[:2]
            # Add small margin to avoid edge issues
            margin = int(min(h, w) * 0.02)
            contour = np.array([
                [margin, margin], 
                [w-margin, margin], 
                [w-margin, h-margin], 
                [margin, h-margin]
            ], dtype=np.int32)
            
        warped = perspective_transform(gray, contour)
        side = warped.shape[0]
        
        # Try to detect actual grid lines for better cell extraction
        h_lines, v_lines = detect_grid_lines(warped)
        
        board = []; confidences = []
        
        for i in range(9):
            row = []; conf_row = []
            for j in range(9):
                try:
                    # Use detected grid lines if available, otherwise use equal division
                    if h_lines and v_lines and len(h_lines) >= 10 and len(v_lines) >= 10:
                        y1, y2 = h_lines[i], h_lines[i+1]
                        x1, x2 = v_lines[j], v_lines[j+1]
                    else:
                        y1, y2 = int(i*side/9), int((i+1)*side/9)
                        x1, x2 = int(j*side/9), int((j+1)*side/9)
                    
                    cell = warped[y1:y2, x1:x2]
                    
                    # Try extracting digit with multiple attempts
                    digit_input = extract_digit(cell)
                    
                    if digit_input is not None:
                        # Make prediction
                        pred = model.predict(digit_input, verbose=0)
                        probabilities = pred[0]
                        val = np.argmax(probabilities)
                        conf = probabilities[val]
                        
                        # Get second best prediction for validation
                        sorted_probs = np.sort(probabilities)[::-1]
                        second_best_conf = sorted_probs[1] if len(sorted_probs) > 1 else 0
                        
                        # Calculate confidence gap (higher = more certain)
                        conf_gap = conf - second_best_conf
                        
                        # SMART VALIDATION LOGIC
                        accept = False
                        
                        # Rule 1: Very high confidence (>85%)
                        if conf > 0.85 and val != 0:
                            accept = True
                        
                        # Rule 2: High confidence with good gap (>65% with 20% gap)
                        elif conf > 0.65 and conf_gap > 0.20 and val != 0:
                            accept = True
                        
                        # Rule 3: Medium confidence with large gap (>50% with 30% gap)
                        elif conf > 0.50 and conf_gap > 0.30 and val != 0:
                            accept = True
                        
                        # Rule 4: Lower confidence but dominant (>40% with 35% gap)
                        elif conf > 0.40 and conf_gap > 0.35 and val != 0:
                            accept = True
                        
                        if accept:
                            row.append(int(val))
                            conf_row.append(float(conf))
                        else:
                            row.append(0)
                            conf_row.append(0.0)
                    else:
                        row.append(0)
                        conf_row.append(0.0)
                except Exception as e:
                    print(f"Error processing cell ({i},{j}): {e}")
                    row.append(0); conf_row.append(0.0)
            board.append(row); confidences.append(conf_row)
        
        # Post-processing: Validate detected grid
        total_digits = sum(1 for r in board for c in r if c != 0)
        print(f"Detected {total_digits} digits in grid")
        
        # If too few digits detected, might be detection issue
        if total_digits < 17:
            print("Warning: Very few digits detected. Results may be unreliable.")
            
        # Sanitize duplicates
        board = sanitize_board(board, confidences)
        
        # Auto-Repair Solve Loop
        solved_board = [r[:] for r in board]
        if not solve_sudoku(solved_board):
            print("Initial solve failed. Auto-repairing...")
            for attempt in range(10): # Try removing up to 10 weakest links
                min_conf = 2.0; min_pos = None
                for r in range(9):
                    for c in range(9):
                        if board[r][c] != 0 and confidences[r][c] < min_conf:
                            min_conf = confidences[r][c]; min_pos = (r, c)
                
                if min_pos:
                    r, c = min_pos
                    board[r][c] = 0; confidences[r][c] = 2.0 
                    solved_board = [row[:] for row in board]
                    if solve_sudoku(solved_board):
                        print("Repair successful!")
                        break
                else: break

        return jsonify({"original": board, "solved": solved_board})
    
    except Exception as e:
        print(f"Fatal error in process_sudoku: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)