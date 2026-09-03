# Day 1: Image Processing Foundations & The World of Computer Vision

## Computer Vision — Week Overview

Welcome to the Computer Vision module. Over the next five days, we'll go from understanding how a computer "sees" an image all the way to deploying a working model that can analyze real airport surveillance video.

| Day | Topic | What You'll Learn |
|-----|-------|------------------|
| **Day 1** | Image Processing Foundations & CV Applications | How digital images work, pixels, color spaces, basic operations, and where CV is used in the real world |
| **Day 2** | Core CV Techniques | Edge detection, filtering, thresholding, contour detection, feature extraction with OpenCV |
| **Day 3** | Video Analysis & Object Tracking | Working with video frames, background subtraction, optical flow, object tracking algorithms |
| **Day 4** | Deep Learning for Computer Vision | YOLO for detection, activity recognition, building the airplane turnaround pipeline |
| **Day 5** | Deployment & Project Submission | Packaging your model, building an interface, calculating turnaround time, submitting to the repo |

**Final Project:** Build a Computer Vision model to calculate the turnaround time of an airplane from surveillance video footage. More details at the end of this document.

---

## 1. What is Computer Vision?

### 1.1 The Simple Definition

Computer Vision is the field of artificial intelligence that teaches machines to interpret and understand visual information from the world — images, videos, and live camera feeds.

Think about what your eyes and brain do every second: you look at a scene, instantly recognize objects, understand their positions, read text, detect motion, and make decisions based on what you see. Computer Vision aims to give machines this same ability.

### 1.2 Computer Vision vs Image Processing vs Machine Learning

These three terms are closely related but not the same:

**Image Processing** is about transforming images — making them sharper, removing noise, adjusting colors, extracting edges. The input is an image, and the output is another (improved) image. It's a toolkit of techniques.

**Computer Vision** uses image processing as a foundation but goes further — it tries to *understand* what's in an image. The input is an image, but the output is *meaning*: "there are two cars and a pedestrian in this scene."

**Machine Learning / Deep Learning** provides the learning algorithms that power modern computer vision. Instead of hand-coding rules for recognizing a cat, we show the model thousands of cat images and let it learn the patterns itself.

```
Image Processing:   Image → Transform → Better Image
Computer Vision:    Image → Analyze → Understanding/Meaning
Machine Learning:   Data → Learn Patterns → Predictions
```

In practice, a modern CV system uses all three: image processing to clean and prepare the data, machine learning to recognize patterns, and computer vision principles to interpret the results.

### 1.3 A Brief History

The story of computer vision is the story of teaching machines to see, one small step at a time.

**1966 — The MIT Summer Vision Project**
Professor Marvin Minsky assigned a summer project to a group of students: "build a system that can describe what it sees through a camera." They thought it would take one summer. It took the entire field decades — and we're still working on it.

**1970s — Edge Detection**
Researchers developed mathematical methods to find boundaries and edges in images. The Sobel and Canny edge detectors from this era are still used today.

**1980s–1990s — Feature-Based Recognition**
Instead of trying to understand whole images, researchers focused on finding distinctive features — corners, blobs, patterns — that could identify objects. Algorithms like SIFT (Scale-Invariant Feature Transform) emerged.

**2001 — Viola-Jones Face Detection**
The first real-time face detection algorithm. Suddenly, digital cameras could find faces automatically. This was a breakthrough that brought CV into consumer products.

**2012 — The Deep Learning Revolution (AlexNet)**
A deep convolutional neural network called AlexNet won the ImageNet competition by a massive margin, proving that deep learning could see far better than any hand-crafted algorithm. This moment changed everything.

**2015 — YOLO (You Only Look Once)**
Real-time object detection became possible. A single neural network could identify and locate multiple objects in an image in milliseconds.

**2020s — The Current Era**
Vision Transformers (ViT), foundation models (CLIP, SAM), and multimodal models (GPT-4V) have pushed CV capabilities to near-human (and sometimes superhuman) levels. Computer vision is now a $20+ billion industry deployed in virtually every sector.

---

## 2. How Computers See: Digital Image Fundamentals

### 2.1 What is a Digital Image?

To us, a photograph is a scene — a sunset, a face, a city skyline. To a computer, that same photograph is a giant grid of numbers. Understanding this distinction is the first step in computer vision.

A **digital image** is a rectangular grid (matrix) of tiny squares called **pixels**. Each pixel stores a numerical value that represents color or intensity at that specific point.

When you hear that an image is "1920×1080," that means it has 1,920 columns and 1,080 rows of pixels — over 2 million tiny colored squares working together to form the picture you see.

### 2.2 Pixels — The Atoms of an Image

The word "pixel" comes from "picture element." It's the smallest addressable unit in a digital image.

**Key properties of a pixel:**
- It has a position (row, column) in the grid
- It stores a numerical value representing its color
- On its own, a single pixel is meaningless — meaning emerges from patterns across many pixels

**Grayscale pixels:** A single number from 0 (pure black) to 255 (pure white). The values in between represent shades of gray. A grayscale image is simply a 2D matrix of these numbers.

```
Example: A tiny 4×4 grayscale image

  [  0,  50, 100, 150]     ← Row 0 (dark → lighter)
  [ 50, 100, 150, 200]     ← Row 1
  [100, 150, 200, 250]     ← Row 2
  [150, 200, 250, 255]     ← Row 3 (lighter → white)
```

**Color pixels (RGB):** Instead of one number, each pixel stores three numbers — one for Red, one for Green, one for Blue. Each channel ranges from 0 to 255, giving us 256 × 256 × 256 = **16.7 million possible colors**.

```
A single RGB pixel: (128, 64, 200)
                      ↑    ↑    ↑
                     Red  Green  Blue
                   medium  low   high  → a shade of purple
```

**How computers store color images:**

A color image is stored as a **3D array** with dimensions: Height × Width × Channels.

```
A 1080p color image:
  Shape: (1080, 1920, 3)
         height  width  channels (R, G, B)

  Total values: 1080 × 1920 × 3 = 6,220,800 numbers
```

### 2.3 Image Coordinate System

In image processing, the coordinate system is a bit different from what you learned in math class:

```
(0,0) ──────────────────→ x (columns)
  │
  │     Image
  │     Area
  │
  ↓
  y (rows)
```

- The **origin (0,0)** is at the **top-left** corner (not bottom-left like in math)
- **x** increases to the right (columns)
- **y** increases downward (rows)
- To access a pixel: `image[y, x]` or `image[row, column]`

This is important when you start writing code — getting the axes confused is one of the most common beginner mistakes.

### 2.4 Image Resolution and File Size

**Resolution** refers to the total number of pixels in an image. Higher resolution means more detail but also larger file sizes and more computation.

| Resolution Name | Dimensions | Total Pixels | Approx. File Size (uncompressed) |
|----------------|-----------|-------------|--------------------------------|
| VGA | 640 × 480 | 307K | ~0.9 MB |
| HD (720p) | 1280 × 720 | 921K | ~2.8 MB |
| Full HD (1080p) | 1920 × 1080 | 2.1M | ~6.2 MB |
| 4K (UHD) | 3840 × 2160 | 8.3M | ~24.9 MB |
| 8K | 7680 × 4320 | 33.2M | ~99.5 MB |

**For video:** Multiply by the frame rate. A 1080p video at 30 fps generates 30 × 6.2 MB = **186 MB of raw pixel data per second**. This is why video compression (MP4, H.264) is essential, and why efficient processing matters for real-time CV.

---

## 3. Color Spaces — Different Ways to Describe Color

### 3.1 Why Do We Need Different Color Spaces?

RGB is intuitive for displays, but it's not always the best representation for analysis. Different tasks benefit from different ways of describing color.

Imagine you want to detect all red objects in an image. In RGB, "red" is spread across all three channels — a bright red pixel might be (255, 0, 0), but a dark red could be (150, 20, 20), and an orange-red might be (255, 100, 50). Defining "red" in RGB is messy.

In HSV, "red" is simply a range of Hue values, regardless of how bright or saturated the color is. Much cleaner.

### 3.2 RGB (Red, Green, Blue)

The standard color model for digital displays.

**How it works:** Every color is a mix of Red, Green, and Blue light at different intensities (0–255).

**Strengths:** Direct mapping to how screens display color. Simple to understand.

**Weaknesses:** Not intuitive for describing colors the way humans think about them. Difficult to isolate specific colors for detection tasks.

**Common RGB values:**

| Color | R | G | B |
|-------|---|---|---|
| Black | 0 | 0 | 0 |
| White | 255 | 255 | 255 |
| Pure Red | 255 | 0 | 0 |
| Pure Green | 0 | 255 | 0 |
| Pure Blue | 0 | 0 | 255 |
| Yellow | 255 | 255 | 0 |
| Cyan | 0 | 255 | 255 |
| Orange | 255, | 165 | 0 |

### 3.3 Grayscale

A single-channel representation where each pixel is one value from 0 (black) to 255 (white).

**When to use grayscale:**
- Edge detection (edges are about intensity changes, not color)
- Feature detection (many algorithms work on intensity only)
- Reducing computation (1 channel instead of 3 = 3× faster)
- When color doesn't carry useful information

**Converting RGB to Grayscale:**

The standard formula weights the channels differently because human eyes are more sensitive to green:

```
Gray = 0.299 × R + 0.587 × G + 0.114 × B
```

In OpenCV:
```python
import cv2
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Note: OpenCV loads images in **BGR** order (not RGB) — a common gotcha for beginners.

### 3.4 HSV (Hue, Saturation, Value)

HSV separates color information in a way that's closer to how humans perceive color.

**Hue (0–179 in OpenCV):** The type of color — red, orange, yellow, green, blue, purple. Think of it as the position on a color wheel.

**Saturation (0–255):** How "pure" the color is. 0 = gray (no color), 255 = fully vivid.

**Value (0–255):** How bright the color is. 0 = black, 255 = maximum brightness.

**Why HSV is powerful for computer vision:**
- You can detect specific colors by filtering on Hue alone
- Lighting changes mainly affect Value, not Hue — making detection robust to shadows
- Color-based segmentation becomes much simpler

**Example — Detecting "blue" objects:**
```python
import cv2
import numpy as np

image = cv2.imread('photo.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for blue color
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply mask to original image
result = cv2.bitwise_and(image, image, mask=mask)
```

### 3.5 LAB (Lightness, A, B)

**L:** Lightness (0 = black, 100 = white)
**A:** Green-to-Red axis
**B:** Blue-to-Yellow axis

LAB is designed to be **perceptually uniform** — a change of 1 unit in LAB looks like the same amount of change to human eyes, regardless of where you are in the color space. This makes it ideal for color correction and comparing how "different" two colors look.

### 3.6 When to Use Which Color Space

| Task | Best Color Space | Why |
|------|-----------------|-----|
| Displaying images | RGB | Matches screen output |
| Color-based detection | HSV | Hue isolates color type |
| Edge detection | Grayscale | Only needs intensity |
| Color correction | LAB | Perceptually uniform |
| Shadow-robust detection | HSV | Value channel captures brightness separately |
| Reducing computation | Grayscale | 1 channel instead of 3 |

---

## 4. Basic Image Operations

### 4.1 Reading, Displaying, and Saving Images

```python
import cv2

# Read an image
image = cv2.imread('photo.jpg')          # Loads in BGR format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Image properties
print(f"Shape: {image.shape}")           # (height, width, channels)
print(f"Size: {image.size}")             # total number of values
print(f"Data type: {image.dtype}")       # uint8 (0-255)

# Display
cv2.imshow('Window Title', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('output.jpg', image)
```

### 4.2 Resizing

Changing image dimensions. Essential for preparing images for neural networks (which expect fixed input sizes).

```python
# Resize to specific dimensions
resized = cv2.resize(image, (640, 480))          # (width, height)

# Resize by scale factor
half = cv2.resize(image, None, fx=0.5, fy=0.5)   # 50% of original
double = cv2.resize(image, None, fx=2.0, fy=2.0)  # 200% of original
```

**Interpolation methods:**

| Method | When to Use |
|--------|------------|
| `cv2.INTER_AREA` | Shrinking images (best quality) |
| `cv2.INTER_LINEAR` | Default, works for both enlarge and shrink |
| `cv2.INTER_CUBIC` | Enlarging images (better quality, slower) |

### 4.3 Cropping

Cropping is just array slicing — selecting a rectangular region from the image.

```python
# Crop a region: image[y_start:y_end, x_start:x_end]
face = image[100:300, 150:350]    # 200×200 pixel crop
```

### 4.4 Rotation and Flipping

```python
# Rotate
center = (image.shape[1]//2, image.shape[0]//2)
matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

# Flip
flipped_h = cv2.flip(image, 1)    # Horizontal flip
flipped_v = cv2.flip(image, 0)    # Vertical flip
flipped_both = cv2.flip(image, -1) # Both axes
```

### 4.5 Drawing on Images

Useful for annotating detection results, drawing bounding boxes, and creating visualizations.

```python
# Rectangle (bounding box)
cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Circle
cv2.circle(image, center=(300, 200), radius=50, color=(0, 0, 255), thickness=-1)

# Text
cv2.putText(image, 'Airplane', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Line
cv2.line(image, (0, 0), (500, 500), color=(255, 0, 0), thickness=3)
```

### 4.6 Image Arithmetic

**Brightness adjustment:**
```python
brighter = cv2.add(image, np.ones_like(image) * 50)   # Add 50 to all pixels
darker = cv2.subtract(image, np.ones_like(image) * 50)
```

**Blending two images:**
```python
# Weighted addition: output = α × image1 + β × image2 + γ
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
```

---

## 5. Image Filtering

### 5.1 What is Filtering?

Filtering means applying a mathematical operation to each pixel based on its neighbors. This is done by sliding a small matrix (called a **kernel** or **filter**) across the image and computing a new value for each pixel.

The process is called **convolution** — the same operation used in CNNs, but here we define the kernels manually instead of learning them.

### 5.2 Blurring (Smoothing)

Blurring reduces noise and detail. It's one of the most common preprocessing steps.

**Average Blur:** Replace each pixel with the average of its neighbors.
```python
blurred = cv2.blur(image, (5, 5))   # 5×5 averaging kernel
```

**Gaussian Blur:** Uses a bell-curve-shaped kernel — pixels closer to the center have more influence. Produces more natural-looking blur.
```python
gaussian = cv2.GaussianBlur(image, (5, 5), 0)
```

**Median Blur:** Replaces each pixel with the median of its neighbors. Excellent at removing salt-and-pepper noise while preserving edges.
```python
median = cv2.medianBlur(image, 5)
```

### 5.3 Sharpening

The opposite of blurring — enhances edges and details.

```python
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])
sharpened = cv2.filter2D(image, -1, kernel)
```

### 5.4 Noise Reduction

Real-world images always contain noise from camera sensors, compression, or transmission. Denoising is critical for accurate analysis.

```python
# Non-local means denoising (slow but effective)
denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
```

---

## 6. Computer Vision Applications — Where It's Changing the World

### 6.1 The Scale of Impact

Computer vision is no longer experimental. In 2025, the global CV market reached **$20.75 billion**, and by 2026, enterprises are generating over **55 billion predictions annually** using vision AI. It has moved from laboratory curiosity to critical operational infrastructure.

Let's walk through the major application areas to understand the breadth of what's possible.

### 6.2 Healthcare & Medical Imaging

**What it does:** Analyzes X-rays, CT scans, MRIs, and microscope images to assist doctors in diagnosis.

**Real-world examples:**
- Detecting tumors in mammograms with accuracy matching experienced radiologists
- Identifying diabetic retinopathy from retinal scans (Google's AI achieved specialist-level performance)
- Counting and classifying blood cells automatically
- Monitoring surgical procedures in real-time

**Why it matters:** Radiologist shortage is a global problem. CV systems can screen thousands of scans per hour, flagging suspicious cases for human review.

### 6.3 Autonomous Vehicles & Transportation

**What it does:** Enables vehicles to perceive and navigate the world using cameras and sensors.

**Real-world examples:**
- Tesla Autopilot: 8 cameras providing 360° vision, processing video in real-time
- Traffic sign recognition (reading speed limits, stop signs)
- Pedestrian and cyclist detection for emergency braking
- Lane detection and lane-keeping assist
- Parking space detection and automated parking

**The challenge:** A self-driving car must process millions of pixels per second, detect objects in all weather conditions, and make life-critical decisions in milliseconds.

### 6.4 Manufacturing & Quality Control

**What it does:** Inspects products on assembly lines at superhuman speed and consistency.

**Real-world examples:**
- Detecting microscopic defects in semiconductor chips
- Checking paint quality on automotive body panels
- Verifying correct assembly of electronic components
- Food quality inspection (sorting produce by ripeness, detecting contamination)
- Reading barcodes and QR codes at high speed

**The advantage:** A human inspector gets tired after hours. A CV system maintains the same accuracy at 3 AM as at 9 AM, processing thousands of items per minute.

### 6.5 Retail & E-Commerce

**What it does:** Transforms the shopping experience through visual intelligence.

**Real-world examples:**
- Amazon Go: cashierless stores using cameras to track what you pick up
- Visual product search: take a photo of shoes you like, find similar products online
- Planogram compliance: checking if store shelves are stocked correctly
- Customer analytics: counting foot traffic, analyzing movement patterns
- Virtual try-on: seeing how glasses or clothes look on you through your phone camera

### 6.6 Agriculture

**What it does:** Monitors crops and livestock with precision that manual inspection can't match.

**Real-world examples:**
- Drone-based crop health monitoring using multispectral imaging
- Detecting plant diseases from leaf images before symptoms are visible to humans
- Automated fruit picking robots that identify ripe produce
- Weed detection and precision herbicide application
- Livestock monitoring: detecting illness, counting animals, tracking behavior

### 6.7 Security & Surveillance

**What it does:** Automates monitoring and threat detection across camera networks.

**Real-world examples:**
- Facial recognition for access control (airports, offices)
- Anomaly detection: identifying unusual behavior in crowds
- License plate recognition for parking and toll systems
- Perimeter security: detecting intrusions in restricted areas
- Crowd density estimation for safety management

### 6.8 Aviation & Airport Operations

This is directly relevant to our final project.

**What it does:** Monitors airport operations to improve efficiency, safety, and on-time performance.

**Real-world examples:**
- **Aircraft turnaround monitoring** — tracking all ground operations (fueling, boarding, baggage) from gate cameras
- **Amsterdam Schiphol's "Deep Turnaround"** — AI system providing real-time visibility into turnaround status, improving on-time departure rates
- Runway foreign object debris (FOD) detection
- Passenger flow optimization through terminals
- Automated gate assignment based on aircraft type detection

**Aircraft turnaround time** is the duration between an aircraft arriving at a gate and departing again. It involves dozens of coordinated activities: passenger deboarding, cleaning, catering, fueling, baggage handling, boarding, and pushback. Delays in any step cascade through the airline's schedule. Computer vision can monitor all these activities automatically from existing camera infrastructure.

### 6.9 Sports & Entertainment

**What it does:** Enhances broadcast, coaching, and fan experience.

**Real-world examples:**
- Player tracking and performance analytics (speed, distance, positioning)
- Automated highlight generation from game footage
- Ball trajectory prediction in cricket, tennis, baseball (Hawk-Eye)
- AR/VR experiences: overlaying stats on live broadcast
- Referee assistance: offside detection in football (VAR)

### 6.10 The Common Thread

Across all these applications, computer vision is doing variations of the same fundamental tasks:

| Task | Description | Example |
|------|------------|---------|
| **Image Classification** | What is in this image? | "This X-ray shows pneumonia" |
| **Object Detection** | Where are the objects? | "Car at position (100,200), pedestrian at (400,300)" |
| **Semantic Segmentation** | Label every pixel | "These pixels are road, those are sidewalk" |
| **Instance Segmentation** | Separate individual objects | "Person 1 is here, Person 2 is there" |
| **Object Tracking** | Follow objects across video frames | "This car moved from gate A to runway B" |
| **Action Recognition** | What is happening? | "The ground crew is currently fueling" |
| **Pose Estimation** | Where are body joints? | "The worker is reaching overhead" |

Our airplane turnaround project will use several of these: detection (identifying equipment and vehicles), tracking (following their movements), and potentially action recognition (determining what activity is happening).

---

## 7. OpenCV — Your Primary Tool

### 7.1 What is OpenCV?

OpenCV (Open Source Computer Vision Library) is the world's most widely used computer vision library. Created in 1999 by Intel, it's now maintained by a global community and used by companies from startups to Google.

**Why OpenCV?**
- Over **2,500 optimized algorithms** for image and video analysis
- Runs on Windows, macOS, Linux, Android, iOS
- Interfaces for Python, C++, Java
- Used by **47,000+** companies and organizations worldwide
- Free and open-source (Apache 2.0 license)

### 7.2 Installation

```bash
pip install opencv-python
pip install opencv-contrib-python    # includes extra modules
```

### 7.3 The Essential OpenCV Workflow

Almost every CV project follows this pattern:

```
Read → Preprocess → Analyze → Visualize → Save/Act
```

1. **Read:** Load image or capture video frame
2. **Preprocess:** Resize, convert color space, filter, threshold
3. **Analyze:** Detect edges, find contours, run detector, classify
4. **Visualize:** Draw results (boxes, labels, masks) on the image
5. **Save/Act:** Write output file, trigger alert, send API response

### 7.4 Reading Video (Important for Our Project)

Since our final project works with video, here's how to process video frame-by-frame:

```python
import cv2

# Open video file
cap = cv2.VideoCapture('Airplane_Video.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video: {width}x{height} @ {fps} fps, {duration:.1f} seconds")

# Process frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Your processing here ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1

cap.release()
print(f"Processed {frame_count} frames")
```

---

## 8. Hands-On Exercise: Your First Image Processing Pipeline

Try this exercise to put today's concepts into practice.

```python
import cv2
import numpy as np

# 1. Load an image
image = cv2.imread('sample.jpg')
print(f"Original shape: {image.shape}")
print(f"Pixel at (100, 100): {image[100, 100]}")  # BGR values

# 2. Convert to different color spaces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. Basic operations
resized = cv2.resize(image, (640, 480))
cropped = image[50:250, 100:400]
flipped = cv2.flip(image, 1)

# 4. Filtering
blurred = cv2.GaussianBlur(image, (7, 7), 0)

# 5. Draw annotations
annotated = image.copy()
cv2.rectangle(annotated, (100, 50), (400, 250), (0, 255, 0), 2)
cv2.putText(annotated, 'Region of Interest', (100, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 6. Save results
cv2.imwrite('grayscale.jpg', gray)
cv2.imwrite('blurred.jpg', blurred)
cv2.imwrite('annotated.jpg', annotated)

print("Done! Check the output files.")
```

---

## 9. Final Project Preview: Airplane Turnaround Time Calculator

### What is Turnaround Time?

In aviation, **turnaround time (TAT)** is the time between an aircraft arriving at a gate (wheels stop / in-block) and departing (wheels start / off-block). It's one of the most critical operational metrics for airlines.

**A typical turnaround involves:**
1. Aircraft arrives and parks at gate (in-block)
2. Jet bridge connects / stairs positioned
3. Passengers deplane
4. Cabin cleaning
5. Catering trucks service the aircraft
6. Fueling
7. Baggage unloading / loading
8. Passengers board
9. Jet bridge disconnects
10. Aircraft pushback and departs (off-block)

### Your Task

Given a video of an airplane at a gate (`Airplane_Video.mp4`), build a computer vision system that:

1. **Detects** the aircraft and key ground service equipment (GSE)
2. **Identifies** major turnaround events (arrival, equipment approach, departure)
3. **Calculates** the total turnaround time
4. Presents results in a clear format (timestamps, duration)

### What You'll Build This Week

| Day | What You'll Add |
|-----|----------------|
| Day 1 (today) | Understanding how video frames work, basic image operations |
| Day 2 | Edge detection and feature extraction on video frames |
| Day 3 | Object tracking and motion analysis |
| Day 4 | YOLO-based detection, activity classification |
| Day 5 | Full pipeline assembly, deployment, submission |

### Submission

Your final project will be submitted to a shared repository. Details and the README with exact specifications will be provided.

---

## 10. Key Takeaways from Day 1

1. **A digital image is a grid of numbers.** Grayscale images are 2D matrices; color images are 3D arrays (height × width × 3 channels).

2. **Pixels are the fundamental unit.** Each pixel stores intensity (grayscale) or RGB values (color). Understanding this is understanding how computers "see."

3. **Color spaces matter.** RGB is for displays, HSV is for color detection, Grayscale is for speed and edge detection. Choose the right one for your task.

4. **Image processing is the foundation of computer vision.** Before any AI model can analyze an image, the image often needs preprocessing — resizing, filtering, color conversion, noise removal.

5. **Computer vision is everywhere.** From healthcare to autonomous vehicles to airport operations, the ability to extract meaning from visual data is transforming every industry.

6. **OpenCV is your toolkit.** Learn it well — it's the Swiss Army knife of computer vision and you'll use it throughout this course and your career.

7. **Video is just a sequence of images.** Everything you learn about processing a single image applies to video — you just process it frame by frame.

---

## Preparation for Day 2

**Install these packages if you haven't already:**
```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

**Try the hands-on exercise** in Section 8 with any image from your computer.

**Read (optional):** OpenCV's official Python tutorials at docs.opencv.org

---

**End of Day 1 Lecture Notes**
