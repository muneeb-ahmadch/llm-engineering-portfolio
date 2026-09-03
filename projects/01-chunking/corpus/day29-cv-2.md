# Day 2: Edge Detection, Contour Analysis & Feature Extraction

## Quick Recap from Day 1

Yesterday we learned that a digital image is a grid of numbers, explored color spaces (RGB, Grayscale, HSV), and performed basic operations like resizing, filtering, and blurring. We also introduced the OpenCV workflow: **Read → Preprocess → Analyze → Visualize → Save**.

Today we move into the **Analyze** stage. We'll learn how to extract meaningful structure from images — where the boundaries are, what shapes exist, and what distinctive features can be used to identify and match objects. These are the classical CV techniques that existed long before deep learning and are still essential building blocks in modern pipelines.

**Today's learning path:**

```
Thresholding → Image Gradients → Edge Detection → Morphological Operations → Contour Detection → Feature Extraction
```

Each technique builds on the previous one, and by the end of today you'll be able to look at a raw image and extract structured information from it — outlines of objects, their shapes, areas, and identifiable keypoints.

---

## 1. Image Thresholding — Separating Foreground from Background

### 1.1 What is Thresholding?

Thresholding is the simplest form of image segmentation. It converts a grayscale image into a binary image — every pixel becomes either black (0) or white (255) based on whether it's above or below a threshold value.

Think of it like a pass/fail grade: any pixel with intensity above the threshold "passes" (becomes white), and everything else "fails" (becomes black).

**Why thresholding matters:**
- Simplifies images for analysis (binary is easier to process than 256 shades)
- Separates objects from the background
- Prepares images for contour detection (contours need binary input)
- Used as a preprocessing step in nearly every classical CV pipeline

### 1.2 Simple (Global) Thresholding

A single threshold value is applied to the entire image.

```python
import cv2

image = cv2.imread('photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simple binary threshold
# Every pixel > 127 becomes 255 (white), otherwise 0 (black)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

**The `cv2.threshold()` function returns two values:**
- `ret`: the threshold value used (useful for Otsu's method)
- `binary`: the resulting binary image

**Thresholding types in OpenCV:**

| Type | Rule | Use Case |
|------|------|---------|
| `THRESH_BINARY` | pixel > T → 255, else → 0 | Standard foreground extraction |
| `THRESH_BINARY_INV` | pixel > T → 0, else → 255 | When foreground is darker |
| `THRESH_TRUNC` | pixel > T → T, else → unchanged | Cap bright values |
| `THRESH_TOZERO` | pixel > T → unchanged, else → 0 | Keep only bright regions |
| `THRESH_TOZERO_INV` | pixel > T → 0, else → unchanged | Keep only dark regions |

**The challenge with simple thresholding:** You need to manually choose the right threshold value, and one value rarely works well for the whole image — especially when lighting is uneven.

### 1.3 Otsu's Thresholding — Let the Algorithm Decide

Named after Nobuyuki Otsu (1979), this method **automatically finds the optimal threshold** by analyzing the histogram of the image. It assumes the image has two classes of pixels (foreground and background) and finds the value that minimizes the variance within each class.

```python
# Otsu's method: set threshold to 0, add THRESH_OTSU flag
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's optimal threshold: {ret}")
```

**When Otsu works well:**
- Images with clear bimodal histograms (two distinct peaks — one for foreground, one for background)
- Well-lit images with consistent contrast

**When it struggles:**
- Images with gradual transitions or multiple intensity regions
- Unevenly lit scenes

**Best practice:** Apply Gaussian blur before Otsu's thresholding to smooth out noise:

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 1.4 Adaptive Thresholding — Handling Uneven Lighting

Real-world images almost never have uniform lighting. A page photographed under a desk lamp will be bright on one side and dark on the other. Simple thresholding fails here because no single value works for the entire image.

Adaptive thresholding solves this by computing a **different threshold for each small region** of the image.

```python
# Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,      # method
    cv2.THRESH_BINARY,                # threshold type
    11,                                # block size (neighborhood)
    2                                  # constant subtracted from mean
)

# Adaptive Gaussian Thresholding (weighted by distance)
adaptive_gauss = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
```

**Parameters explained:**
- **Block size (11):** The size of the local neighborhood used to calculate the threshold. Must be odd. Larger = smoother but less detail.
- **C (2):** A constant subtracted from the calculated threshold. Helps fine-tune the result.

**Mean vs Gaussian:**
- **Mean:** Threshold = average of the neighborhood. Simple, fast.
- **Gaussian:** Threshold = weighted average where closer pixels count more. Generally produces better results for text and documents.

### 1.5 Choosing the Right Thresholding Method

| Scenario | Best Method | Why |
|----------|------------|-----|
| Uniform lighting, clear object | Simple binary | Fast, effective |
| Don't know what threshold to use | Otsu's | Automatically optimal |
| Uneven lighting (shadows, lamps) | Adaptive | Local thresholds adapt |
| Document/text extraction | Adaptive Gaussian | Handles page curvature and shadows |
| Preprocessing for contours | Otsu's + blur | Clean binary for findContours |

---

## 2. Image Gradients — The Mathematics of Change

### 2.1 What is an Image Gradient?

An image gradient measures how quickly pixel intensity changes from one location to the next. Where intensity changes sharply — that's an edge. Where it changes gradually — that's a smooth region.

Imagine walking across a flat field and suddenly reaching a cliff. The gradient at the cliff is steep (large change over short distance). Edges in images work the same way — they're locations where pixel values change rapidly.

**A gradient has two properties:**
- **Magnitude:** How strong is the change? (high magnitude = strong edge)
- **Direction:** Which way is the change pointing? (perpendicular to the edge)

### 2.2 Sobel Operator

The Sobel operator uses a pair of 3×3 kernels to compute gradients in the horizontal (x) and vertical (y) directions separately.

**Sobel X kernel (detects vertical edges):**
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

**Sobel Y kernel (detects horizontal edges):**
```
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```

Think of it this way: Sobel X looks for changes left-to-right (so it finds vertical boundaries), and Sobel Y looks for changes top-to-bottom (so it finds horizontal boundaries).

```python
import cv2
import numpy as np

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gradient in X direction (vertical edges)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Gradient in Y direction (horizontal edges)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Combined magnitude
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.uint8(np.clip(magnitude, 0, 255))

# Or use absolute values
abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)
combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
```

**Why `cv2.CV_64F`?** The gradient can be negative (transition from white to black). Using `uint8` (0–255) would clip negative values to 0, losing half the edges. Using 64-bit float preserves both positive and negative transitions.

### 2.3 Laplacian Operator

While Sobel computes first-order derivatives (rate of change), the Laplacian computes the **second-order derivative** (rate of change of the rate of change). It detects edges in all directions simultaneously.

```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)
```

**Comparison:**

| Property | Sobel | Laplacian |
|----------|-------|-----------|
| Order | 1st derivative | 2nd derivative |
| Direction | X or Y separately | All directions at once |
| Noise sensitivity | Moderate | High (needs more blur) |
| Output | Directional edges | All edges |
| Speed | Fast | Fast |

### 2.4 Scharr Operator

A refined version of Sobel with better rotational symmetry. Use when you need more accurate gradients, especially at the cost of slightly more computation.

```python
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
```

---

## 3. Edge Detection — Finding Boundaries

### 3.1 Why Edge Detection Matters

Edges are the most information-rich parts of an image. They mark the boundaries between objects, the outlines of shapes, and the transitions between regions. A human-drawn sketch captures the essence of a scene using only edges — our brains fill in the rest. Edge detection gives computers a similar compressed representation.

**Edges occur where:**
- Objects meet the background
- One surface ends and another begins
- Shadows fall on surfaces
- Textures change

### 3.2 Canny Edge Detection — The Gold Standard

The Canny edge detector, developed by John Canny in 1986, is the most widely used edge detection algorithm. It's considered optimal because it achieves three goals simultaneously:

1. **Low error rate:** Detect as many real edges as possible, with few false positives
2. **Good localization:** Detected edges should be as close to real edges as possible
3. **Minimal response:** Each edge should be marked only once (thin, single-pixel edges)

#### The Four Steps of Canny

**Step 1: Gaussian Smoothing**
First, blur the image with a 5×5 Gaussian kernel to reduce noise. Without this step, noise would create thousands of false edges.

**Step 2: Gradient Computation**
Apply Sobel operators in X and Y to find gradient magnitude and direction at every pixel. Strong gradients = likely edges.

**Step 3: Non-Maximum Suppression (NMS)**
This is the step that makes edges thin. For each pixel on a potential edge, check if it's the local maximum in the gradient direction. If a pixel isn't stronger than its neighbors along the gradient, it's suppressed (set to 0). The result: thick edge regions become thin, single-pixel lines.

**Step 4: Hysteresis Thresholding**
Two thresholds are used instead of one:
- **High threshold:** Pixels above this are definitely edges ("sure edges")
- **Low threshold:** Pixels below this are definitely not edges
- **Between:** These pixels are edges only if they're connected to a sure edge

This two-threshold approach is the secret to Canny's robustness — it keeps weak edges that are part of real boundaries while discarding isolated noise.

#### Canny in OpenCV

```python
import cv2

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional but recommended: blur first
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
```

**Choosing thresholds:**
- Canny recommended a ratio of **2:1 or 3:1** between high and low thresholds
- Common starting points: (50, 150), (100, 200), (30, 90)
- Lower thresholds → more edges (including noise)
- Higher thresholds → fewer edges (only strong ones)

#### Auto-Canny: Automatic Threshold Selection

A practical technique that uses the median pixel intensity to set thresholds automatically:

```python
def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)

edges = auto_canny(blurred)
```

### 3.3 Edge Detection Comparison

| Technique | Strengths | Weaknesses | Best For |
|-----------|----------|-----------|---------|
| **Sobel** | Directional, fast | Thick edges, noise-sensitive | Finding gradients, preprocessing |
| **Laplacian** | All-direction, fast | Very noise-sensitive | Quick all-edge detection |
| **Canny** | Thin edges, robust, two thresholds | Needs parameter tuning | Final edge detection (standard choice) |

**Rule of thumb:** Almost always start with Canny. Use Sobel when you need directional edge information. Use Laplacian only on pre-smoothed images.

---

## 4. Morphological Operations — Cleaning Binary Images

### 4.1 What Are Morphological Operations?

After thresholding or edge detection, binary images often have imperfections: small noise dots, holes inside objects, broken edges, or thin connections between separate objects. Morphological operations clean these up using a **structuring element** (a small shape, usually a rectangle or ellipse) that slides over the image.

Think of them as digital sculpting tools for binary images — you can erode material away, add material, or smooth the surface.

### 4.2 Structuring Elements (Kernels)

```python
import cv2
import numpy as np

# Rectangular kernel
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Elliptical kernel (most common for general use)
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cross-shaped kernel
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

**Which kernel shape?**
- **Rectangle:** General purpose, treats all directions equally
- **Ellipse:** Smoother results, better for round objects
- **Cross:** Useful for connecting only vertically/horizontally adjacent pixels

### 4.3 The Four Core Operations

#### Erosion — Shrink the Foreground

Erosion slides the kernel over the image. A pixel stays white only if **all pixels under the kernel are white**. Otherwise, it becomes black. This shrinks white regions and removes small noise.

```python
eroded = cv2.erode(binary, kernel, iterations=1)
```

**Effects:**
- Removes small white noise dots
- Shrinks foreground objects
- Separates loosely connected objects
- Removes thin protrusions

#### Dilation — Expand the Foreground

The opposite of erosion. A pixel becomes white if **any pixel under the kernel is white**. This grows white regions and fills small holes.

```python
dilated = cv2.dilate(binary, kernel, iterations=1)
```

**Effects:**
- Fills small holes inside objects
- Connects nearby objects
- Thickens edges
- Expands foreground regions

#### Opening — Erosion Followed by Dilation

Remove noise without shrinking the object (erosion removes noise, dilation restores size).

```python
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

**Think of it as:** "Open the door, sweep out the noise, close the door." The object returns to roughly its original size, but the noise is gone.

#### Closing — Dilation Followed by Erosion

Fill holes and close gaps without expanding the object (dilation fills holes, erosion restores size).

```python
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

**Think of it as:** "Close all the holes, then trim back to size."

### 4.4 Advanced Morphological Operations

```python
# Morphological Gradient: dilation - erosion = object outline
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# Top Hat: original - opening = bright spots smaller than kernel
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)

# Black Hat: closing - original = dark spots smaller than kernel
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
```

### 4.5 Morphological Operations Summary

| Operation | Formula | Purpose | Use When |
|-----------|---------|---------|---------|
| **Erosion** | Min filter | Shrink, remove noise | Small white specks in black area |
| **Dilation** | Max filter | Expand, fill holes | Small black holes in white area |
| **Opening** | Erode → Dilate | Remove noise, keep size | Noise removal on objects |
| **Closing** | Dilate → Erode | Fill holes, keep size | Closing gaps inside objects |
| **Gradient** | Dilate - Erode | Edge outline | Finding object boundaries |

### 4.6 Practical Example: Cleaning Up a Thresholded Image

```python
import cv2
import numpy as np

# Read and threshold
image = cv2.imread('photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean up with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Step 1: Remove noise (opening)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Step 2: Fill holes (closing)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

# Now 'cleaned' is ready for contour detection
```

---

## 5. Contour Detection — Finding and Analyzing Shapes

### 5.1 What is a Contour?

A contour is a curve that joins all continuous points along a boundary that share the same color or intensity. In simpler terms: it's the outline of a shape in a binary image.

If thresholding and edge detection help us find *where* things change, contour detection gives us *organized shapes* — closed boundaries we can measure, classify, and reason about.

**Contours vs Edges:**
- **Edges:** Individual pixels where intensity changes. Can be disconnected fragments.
- **Contours:** Organized, connected curves forming closed (or open) boundaries. Stored as ordered lists of points.

### 5.2 Finding Contours

```python
import cv2

# Contours need binary input — always threshold first
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,           # retrieval mode
    cv2.CHAIN_APPROX_SIMPLE      # approximation method
)

print(f"Found {len(contours)} contours")

# Draw all contours on the original image
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
```

**Retrieval modes:**

| Mode | What It Retrieves | When to Use |
|------|------------------|-------------|
| `RETR_EXTERNAL` | Only outermost contours | When you want objects, not holes |
| `RETR_LIST` | All contours, flat list | Simple collection of all shapes |
| `RETR_TREE` | All contours with hierarchy | When nesting matters (holes in objects) |
| `RETR_CCOMP` | Two-level hierarchy | Object + hole level only |

**Approximation methods:**

| Method | What It Stores | Size |
|--------|---------------|------|
| `CHAIN_APPROX_NONE` | Every point on the contour | Large |
| `CHAIN_APPROX_SIMPLE` | Only endpoints of line segments | Compact (use this) |

### 5.3 Contour Properties — Measuring Shapes

Once you have contours, you can extract rich information about each shape.

#### Area and Perimeter

```python
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    print(f"Area: {area:.0f} px², Perimeter: {perimeter:.1f} px")
```

#### Bounding Box

```python
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Aspect ratio
aspect_ratio = w / h
```

#### Minimum Enclosing Circle

```python
(cx, cy), radius = cv2.minEnclosingCircle(contour)
cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
```

#### Centroid (Center of Mass)

```python
M = cv2.moments(contour)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
```

#### Contour Approximation (Simplifying Shapes)

The Douglas-Peucker algorithm reduces the number of points in a contour while preserving its shape.

```python
epsilon = 0.02 * cv2.arcLength(contour, True)   # 2% of perimeter
approx = cv2.approxPolyDP(contour, epsilon, True)

# Number of vertices tells you the shape
vertices = len(approx)
if vertices == 3:
    shape = "Triangle"
elif vertices == 4:
    shape = "Rectangle"
elif vertices > 8:
    shape = "Circle"
```

#### Convex Hull and Convexity Defects

```python
hull = cv2.convexHull(contour)
cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)

# Solidity: how "filled" the shape is
hull_area = cv2.contourArea(hull)
solidity = area / hull_area   # 1.0 = convex, < 1.0 = concave
```

### 5.4 Filtering Contours

In practice, you'll find hundreds of contours. Filtering is essential:

```python
# Filter by area (remove noise)
min_area = 500
large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Filter by aspect ratio (find elongated objects)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 0.8 < w/h < 1.2:  # roughly square
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

# Sort by area (largest first)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest = sorted_contours[0]
```

### 5.5 Complete Contour Analysis Pipeline

```python
import cv2
import numpy as np

# Load and preprocess
image = cv2.imread('photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Clean with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Analyze each contour
result = image.copy()
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area < 500:
        continue

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

    # Label
    label = f"#{i} A={area:.0f}"
    cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite('contour_analysis.jpg', result)
```

---

## 6. Feature Detection — Finding Distinctive Points

### 6.1 What are Features?

In everyday life, you recognize a building by its distinctive elements — a unique window shape, an unusual door, a distinctive corner. Computer vision does the same thing. A **feature** is a distinctive, identifiable point or region in an image that can be reliably found again in other images of the same scene.

**Good features are:**
- **Repeatable:** Found in the same location every time
- **Distinctive:** Different from their surroundings
- **Invariant:** Recognizable despite changes in scale, rotation, or lighting

### 6.2 Why Features Matter

Features enable some of the most important CV tasks:
- **Image matching:** Find the same object in different images
- **Panorama stitching:** Align and merge overlapping photos
- **Object recognition:** Identify known objects by their features
- **Motion tracking:** Follow features across video frames
- **3D reconstruction:** Build 3D models from 2D images
- **Image registration:** Align images from different sensors or times

For our airplane project, features can help us track specific parts of the aircraft and ground equipment across frames.

### 6.3 Harris Corner Detection

Corners are excellent features because they're distinctive in all directions. A flat region looks the same everywhere. An edge looks the same along its length. But a corner is unique — it changes in every direction you move.

**How Harris works:**

The algorithm places a small window on the image and slides it in all directions:
- **Flat region:** No change in any direction → not a feature
- **Edge:** Change in one direction only → weak feature
- **Corner:** Change in all directions → strong feature

```python
import cv2
import numpy as np

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection
# Parameters: image, blockSize, ksize, k
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate to make corners visible
corners = cv2.dilate(corners, None)

# Mark corners on image (where response is above 1% of max)
result = image.copy()
result[corners > 0.01 * corners.max()] = [0, 0, 255]
```

**Parameters:**
- **blockSize:** Size of the neighborhood window (2–7)
- **ksize:** Aperture parameter for Sobel (3, 5, 7)
- **k:** Harris detector free parameter (typically 0.04–0.06)

**Harris is good for:** finding corners in architectural images, chessboards, or structured scenes. But it's not scale-invariant — a corner at one zoom level might not be detected at another.

### 6.4 SIFT (Scale-Invariant Feature Transform)

SIFT, introduced by David Lowe in 2004, was a breakthrough because it finds features that are **invariant to scale and rotation**. The same keypoint can be found whether the image is zoomed in, zoomed out, or rotated.

#### How SIFT Works (4 Steps)

**Step 1: Scale-Space Extrema Detection**
Create multiple versions of the image at different blur levels (scales). Compute the Difference of Gaussian (DoG) between consecutive blur levels. Keypoints are found where the DoG response is a local extremum compared to its 26 neighbors (8 spatial + 9 above + 9 below).

**Step 2: Keypoint Localization**
Refine keypoint positions to sub-pixel accuracy. Discard low-contrast keypoints and edge responses (which aren't true corners).

**Step 3: Orientation Assignment**
For each keypoint, compute gradient orientations in its neighborhood and assign a dominant orientation. This makes the descriptor rotation-invariant.

**Step 4: Descriptor Generation**
Create a 128-dimensional vector describing the local image patch around each keypoint. This descriptor is used for matching.

```python
# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

print(f"Found {len(keypoints)} keypoints")
print(f"Descriptor shape: {descriptors.shape}")  # (N, 128)

# Draw keypoints
result = cv2.drawKeypoints(image, keypoints, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

**Each keypoint contains:**
- Position (x, y)
- Scale (size)
- Orientation (angle)
- Response (strength)

### 6.5 ORB (Oriented FAST and Rotated BRIEF)

ORB was developed by OpenCV Labs in 2011 as a **free, fast alternative** to SIFT and SURF (which were patented at the time). It's much faster than SIFT while maintaining good matching performance.

**How ORB works:**
- **Detection:** Uses FAST (Features from Accelerated Segment Test) to find keypoints — much faster than SIFT's DoG approach
- **Orientation:** Computes orientation using intensity centroid
- **Description:** Uses BRIEF (Binary Robust Independent Elementary Features) — a binary descriptor that's faster to compute and match than SIFT's float descriptors

```python
# Create ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect and compute
keypoints, descriptors = orb.detectAndCompute(gray, None)

print(f"Found {len(keypoints)} keypoints")
print(f"Descriptor shape: {descriptors.shape}")  # (N, 32) — binary, compact!

# Draw
result = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
```

### 6.6 Feature Comparison

| Property | Harris | SIFT | ORB |
|----------|--------|------|-----|
| **Type** | Corner only | Keypoint + descriptor | Keypoint + descriptor |
| **Scale invariant** | No | Yes | Partial |
| **Rotation invariant** | No | Yes | Yes |
| **Speed** | Fast | Slow | Very fast |
| **Descriptor size** | None | 128 floats | 32 bytes (binary) |
| **Best for** | Simple corners | Accurate matching | Real-time applications |
| **License** | Free | Free (since 2020) | Free |

### 6.7 Feature Matching Between Images

Once you have descriptors, you can match features between two images:

```python
# Detect features in both images
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match using Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 20 matches
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

**Matcher types:**
- **BFMatcher (Brute-Force):** Compares every descriptor in image 1 with every descriptor in image 2. Thorough but slow.
- **FLANN (Fast Library for Approximate Nearest Neighbors):** Uses tree-based algorithms for faster matching. Better for large feature sets.

---

## 7. Histograms — Understanding Image Statistics

### 7.1 What is a Histogram?

An image histogram is a graph showing how pixel intensities are distributed. The x-axis represents intensity values (0–255), and the y-axis shows how many pixels have that intensity.

Histograms tell you about the overall brightness, contrast, and exposure of an image at a glance.

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plot
plt.figure(figsize=(10, 4))
plt.plot(hist, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.show()
```

### 7.2 Reading Histograms

| Histogram Shape | Meaning | Example |
|----------------|---------|---------|
| Peak at left (low values) | Dark/underexposed image | Night scene |
| Peak at right (high values) | Bright/overexposed image | Snow, sky |
| Spread evenly | Good contrast | Well-exposed photo |
| Two peaks (bimodal) | Two distinct regions | Object + background |
| Narrow peak | Low contrast | Foggy, hazy scene |

### 7.3 Histogram Equalization

Stretches the histogram to use the full 0–255 range, improving contrast.

```python
# Global histogram equalization
equalized = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) — better results
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(gray)
```

**CLAHE** is the recommended approach because it works on small tiles rather than the whole image, preventing over-amplification of noise in homogeneous regions.

---

## 8. Putting It Together — The Analysis Pipeline

Here's the complete pipeline that connects everything we learned today:

```
Grayscale → Blur → Threshold (or Edge Detect) → Morphology → Contours → Measure
```

### Complete Example: Detecting and Measuring Objects

```python
import cv2
import numpy as np

# Load
image = cv2.imread('objects.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess (Day 1 skills)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold (today's skills)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphology (today's skills)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

# Contours (today's skills)
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Analyze and annotate
result = image.copy()
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 1000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # Classify shape
    if circularity > 0.8:
        shape = "Circle"
        color = (0, 255, 0)
    elif 0.6 < circularity < 0.8:
        shape = "Ellipse"
        color = (255, 165, 0)
    else:
        shape = "Polygon"
        color = (0, 0, 255)

    cv2.drawContours(result, [cnt], -1, color, 2)
    cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)
    label = f"{shape} A={area:.0f}"
    cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite('analysis_result.jpg', result)
print(f"Detected {len([c for c in contours if cv2.contourArea(c) > 1000])} objects")
```

---

## 9. Connection to Our Airplane Project

Every technique we learned today will be used in the turnaround time calculator:

| Technique | Project Application |
|-----------|-------------------|
| **Thresholding** | Separating aircraft silhouette from the tarmac background |
| **Edge detection** | Finding the outline of the aircraft, jet bridges, and equipment |
| **Morphological ops** | Cleaning up noisy detections from real camera footage |
| **Contour analysis** | Measuring the position and size of detected objects |
| **Feature detection** | Tracking specific parts of the aircraft across video frames |
| **Histograms** | Detecting lighting changes that indicate time of day or activity |

**Tomorrow (Day 3):** We'll apply these techniques to video — frame differencing, background subtraction, optical flow, and object tracking algorithms. That's when the static analysis from today becomes dynamic motion detection.

---

## 10. Key Takeaways from Day 2

1. **Thresholding converts grayscale to binary.** Use Otsu's when you don't know the threshold, adaptive when lighting is uneven.

2. **Gradients measure intensity change.** Sobel gives directional gradients; edges exist where gradients are strong.

3. **Canny is the gold standard for edge detection.** Its four-step process (blur → gradient → NMS → hysteresis) produces clean, thin, connected edges.

4. **Morphological operations clean binary images.** Opening removes noise, closing fills holes. Always apply after thresholding.

5. **Contours give you shapes.** From a binary image, `findContours` extracts organized boundaries you can measure — area, perimeter, bounding box, centroid.

6. **Features are distinctive points.** Harris finds corners, SIFT finds scale-invariant keypoints, ORB does it fast and free.

7. **The pipeline is sequential.** Grayscale → Blur → Threshold → Morphology → Contours → Measure. Each step feeds the next.

---

## Preparation for Day 3

**Practice:** Run the complete pipeline from Section 8 on different images. Try changing threshold values and morphological kernel sizes to see their effects.

**Think ahead:** Tomorrow we work with video. Start thinking about what changes between consecutive frames when a plane arrives versus when ground equipment approaches.

---

**End of Day 2 Lecture Notes**
