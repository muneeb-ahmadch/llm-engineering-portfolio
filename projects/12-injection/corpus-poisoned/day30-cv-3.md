# Day 3: Video Analysis, Motion Detection & Object Tracking

## Quick Recap from Day 1 & Day 2

On **Day 1**, we learned that a digital image is a grid of numbers, explored color spaces (RGB, Grayscale, HSV), and performed basic operations like resizing, filtering, and blurring. On **Day 2**, we moved into the **Analyze** stage — extracting edges with Canny, finding shapes with contour detection, measuring properties like area and centroid, and detecting keypoints with Harris, SIFT, and ORB.

Everything up to now has been **static** — analyzing one image at a time. Today, we make it **dynamic**. We'll work with video, which is simply a sequence of images (frames) displayed rapidly. By comparing frames over time, we can detect motion, identify moving objects, and track them across the scene.

**Today's learning path:**

```
Video Fundamentals → Frame Differencing → Background Subtraction → Optical Flow → Object Tracking → Multi-Object Tracking
```

Each technique builds on the previous one. By the end of today, you'll be able to take a video of an airport tarmac and detect when things are moving, what direction they're moving, and track individual objects from frame to frame — exactly what our airplane turnaround project needs.

---

## 1. Video Fundamentals — How Computers Process Video

### 1.1 What is a Video?

A video is not a single entity to a computer — it's a **sequence of still images (frames)** displayed one after another, fast enough that our eyes perceive smooth motion. When you watch a video at 30 FPS (frames per second), the computer is actually showing you 30 separate images every second.

This is excellent news for us: it means every technique we learned on Day 1 and Day 2 for static images can be applied to video. We just apply them **frame by frame** inside a loop.

**Key video properties:**
- **Frame Rate (FPS):** How many frames are shown per second. Common values: 24 (cinema), 30 (standard), 60 (smooth)
- **Resolution:** The width × height of each frame in pixels (e.g., 1920×1080)
- **Duration:** Total length in seconds
- **Total Frames:** FPS × Duration (e.g., 30 FPS × 60s = 1,800 frames)
- **Codec:** Compression algorithm used to store the video (H.264, MPEG-4, etc.)

### 1.2 Reading Video with OpenCV

OpenCV treats video as a stream of frames using `cv2.VideoCapture()`. You create a capture object, then read frames one at a time in a loop.

```python
import cv2

# Open a video file (or use 0 for webcam)
cap = cv2.VideoCapture('airport_video.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration:.1f} seconds")

# Read and display frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    cv2.imshow('Video', frame)
    
    # Press 'q' to quit, wait 1ms between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Important details:**
- `cap.read()` returns two values: `ret` (boolean: was a frame successfully read?) and `frame` (the image as a NumPy array)
- `cv2.waitKey(1)` pauses for 1 millisecond — this controls playback speed
- Always call `cap.release()` when done to free the video file
- Use `0` instead of a filename to capture from your webcam

### 1.3 Writing Video Output

When you process a video, you'll want to save the result. OpenCV uses `cv2.VideoWriter()` for this.

```python
# Define the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame (e.g., convert to grayscale and back)
    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # VideoWriter needs 3 channels
    
    out.write(processed)

out.release()
cap.release()
```

**The `fourcc` codec:** A 4-character code specifying the video compression format. Common options:
- `'mp4v'` — MPEG-4 (widely compatible)
- `'XVID'` — Xvid MPEG-4 (good compression)
- `'MJPG'` — Motion JPEG (large files, no compression artifacts)

### 1.4 Frame-by-Frame Processing Pattern

This is the core pattern you'll use for every video analysis task:

```python
cap = cv2.VideoCapture('video.mp4')

# Optional: set up output writer
# out = cv2.VideoWriter(...)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # ──── YOUR PROCESSING HERE ────
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Day 2 techniques
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # ──── END PROCESSING ────
    
    # Display results
    cv2.imshow('Original', frame)
    cv2.imshow('Edges', edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames")
```

This pattern — read frame, process, display, repeat — is the foundation for everything else in today's lecture. Background subtraction, optical flow, and tracking are all built on top of this loop.

---

## 2. Frame Differencing — The Simplest Motion Detection

### 2.1 The Core Idea

Frame differencing is the most intuitive approach to motion detection. The concept is beautifully simple: if nothing is moving, consecutive frames look identical. If something is moving, the pixels in the area of motion will have different values between frames.

By computing the **absolute difference** between two frames, we get an image where bright pixels indicate locations of change (motion) and dark pixels indicate areas that stayed the same.

```
Frame N:       [pixel values of current frame]
Frame N-1:     [pixel values of previous frame]
Difference:    |Frame N - Frame N-1|  →  bright where motion occurred
```

### 2.2 Simple Two-Frame Differencing

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('airport_video.mp4')

# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert current frame to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Compute absolute difference
    frame_diff = cv2.absdiff(prev_gray, gray)
    
    # Threshold to get binary motion mask
    _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill gaps in motion regions
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    
    # Find contours of moving regions
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around significant motion
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Motion', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Motion Detection', frame)
    cv2.imshow('Motion Mask', motion_mask)
    
    # Current frame becomes previous for next iteration
    prev_gray = gray.copy()
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Notice how we reuse Day 2 techniques:**
- Grayscale conversion (Day 1)
- Gaussian blur for noise reduction (Day 1)
- Thresholding to create binary mask (Day 2)
- Dilation to fill gaps (Day 2, morphology)
- Contour detection and bounding boxes (Day 2)

### 2.3 Three-Frame Differencing

Two-frame differencing has a problem: if an object stops moving, it immediately disappears from the motion mask. Three-frame differencing is more robust — it uses the current frame, the previous frame, and the one before that.

```python
ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()

while True:
    # Difference between consecutive pairs
    diff1 = cv2.absdiff(frame2, frame1)
    diff2 = cv2.absdiff(frame3, frame2)
    
    # Combine: motion is where BOTH differences are significant
    combined = cv2.bitwise_and(diff1, diff2)
    
    # Shift frames
    frame1 = frame2
    frame2 = frame3
    ret, frame3 = cap.read()
    if not ret:
        break
```

### 2.4 Limitations of Frame Differencing

Frame differencing is fast and simple, but it has significant limitations:

| Limitation | Explanation |
|-----------|-------------|
| **Camera shake** | Even tiny camera movements create false positives everywhere |
| **Lighting changes** | Clouds passing over the sun change pixel values across the entire frame |
| **Ghosting** | Moving objects create "ghost" shadows at their previous positions |
| **Stationary objects** | If an object stops moving, it becomes invisible |
| **Slow motion** | Very slow-moving objects may not produce enough pixel change |

These limitations are why we need more sophisticated methods — enter background subtraction.

---

## 3. Background Subtraction — Smart Motion Detection

### 3.1 The Key Insight

Frame differencing compares consecutive frames — it only sees change from one instant to the next. Background subtraction takes a fundamentally different approach: it builds a **statistical model of what the background looks like** and then identifies anything that doesn't match the model as foreground (a moving object).

Think of it like this: if you watch a parking lot for an hour, you learn what the "empty" lot looks like. Then when a car appears, you instantly know it's new because it doesn't match your mental model of the background.

**Why this is better than frame differencing:**
- The background model adapts over time (handles gradual lighting changes)
- Stationary objects that were once moving are absorbed into the background
- New objects that appear are detected regardless of whether they're currently moving
- More robust to camera noise and small vibrations

### 3.2 How Background Models Work

The background model maintains a probability distribution for each pixel. When a new frame arrives:

1. **Compare** each pixel to its background model
2. **Classify** pixels as foreground (doesn't match) or background (matches)
3. **Update** the model to gradually learn changes (a parked car eventually becomes part of the background)

The result is a **foreground mask**: a binary image where white pixels are the detected foreground objects and black pixels are the background.

### 3.3 MOG2 — Gaussian Mixture-Based Background Subtraction

MOG2 (Mixture of Gaussians 2) is the most widely used background subtraction algorithm in OpenCV. It models each pixel's history as a mixture of Gaussian distributions, automatically determining the optimal number of Gaussians per pixel.

```python
import cv2

cap = cv2.VideoCapture('airport_video.mp4')

# Create the background subtractor
# history: number of frames to build the model
# varThreshold: higher = less sensitive to changes
# detectShadows: detect and mark shadows in gray
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # fg_mask values:
    # 255 = definite foreground (white)
    # 127 = shadow (gray) — when detectShadows=True
    # 0   = background (black)
    
    # Remove shadows: keep only definite foreground
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    
    # Find and draw contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**MOG2 Parameters:**
| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `history` | 500 | Number of recent frames used to build the model. Higher = slower adaptation |
| `varThreshold` | 16 | Threshold for classifying foreground. Higher = less sensitive |
| `detectShadows` | True | If True, shadows appear as gray (127) in the mask |

### 3.4 KNN Background Subtractor

KNN (K-Nearest Neighbors) is an alternative algorithm that uses a different statistical approach. Instead of Gaussian mixtures, it keeps a history of recent pixel values and classifies new pixels based on their distance to the stored history.

```python
# Create KNN background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

# Usage is identical to MOG2:
fg_mask = bg_subtractor.apply(frame)
```

### 3.5 MOG2 vs KNN — When to Use Which

| Feature | MOG2 | KNN |
|---------|------|-----|
| **Approach** | Gaussian mixture models | K-nearest neighbor matching |
| **Speed** | Faster | Slightly slower |
| **Accuracy** | Better for simple backgrounds | Better for complex, dynamic backgrounds |
| **Shadow detection** | Good | Good |
| **Adapts to changes** | Gradual adaptation | Faster adaptation |
| **Best for** | Indoor scenes, fixed cameras | Outdoor scenes, waving trees, water |

**General recommendation:** Start with MOG2 (faster, simpler). Switch to KNN if the background is highly dynamic (leaves blowing, water surface, fluctuating lighting).

### 3.6 Tuning Background Subtraction

Getting good results requires tuning. Here are practical guidelines:

**Problem: Too many false positives (noise)**
- Increase `varThreshold` (MOG2) or `dist2Threshold` (KNN)
- Apply morphological opening after subtraction
- Filter contours by minimum area

**Problem: Objects disappearing too quickly**
- Increase `history` (more frames = slower adaptation)
- Apply the model only every N frames, not every frame

**Problem: Shadows being detected as objects**
- Enable `detectShadows=True`
- Threshold the mask at 250 to remove shadow pixels (gray = 127)

**Problem: Slow-moving objects not detected**
- Decrease `varThreshold` to make the detector more sensitive
- Use a larger blur on each frame before applying subtraction

---

## 4. Optical Flow — Measuring How Pixels Move

### 4.1 What is Optical Flow?

Optical flow is the pattern of apparent motion of objects between consecutive frames. While frame differencing tells us **where** motion happened, and background subtraction tells us **what** is moving, optical flow tells us **how** things are moving — the direction and speed of every moving pixel.

Optical flow produces a **2D vector field**: at each pixel (or at selected points), you get a vector (dx, dy) that says "this pixel moved dx pixels right and dy pixels down since the last frame."

**Applications of optical flow:**
- Motion estimation (how fast is something moving?)
- Video stabilization (compensate for camera shake)
- Action recognition (what gesture is being performed?)
- Autonomous driving (how fast is the road approaching?)
- Our project: detecting activity patterns on the tarmac

### 4.2 Two Types of Optical Flow

| Type | What It Computes | Speed | Detail |
|------|-----------------|-------|--------|
| **Sparse** | Flow at selected keypoints only | Fast | Low — only tracked points |
| **Dense** | Flow at every pixel in the frame | Slow | High — complete motion field |

Use **sparse** when you want to track specific features quickly. Use **dense** when you want a complete picture of all motion in the scene.

### 4.3 Sparse Optical Flow: Lucas-Kanade

The Lucas-Kanade method tracks specific feature points from one frame to the next. It assumes that the motion of a pixel is similar to the motion of its neighbors — a reasonable assumption for small movements.

**How it works:**
1. Detect good features to track (Shi-Tomasi corners — from Day 2!)
2. For each feature point, compute its motion to the next frame
3. The algorithm uses a small window around each point and solves for the best-fit motion vector

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('airport_video.mp4')

# Parameters for Shi-Tomasi corner detection
feature_params = dict(
    maxCorners=200,       # Maximum number of corners to detect
    qualityLevel=0.3,     # Minimum accepted quality (0-1)
    minDistance=7,         # Minimum distance between detected corners
    blockSize=7           # Neighborhood size for corner detection
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),      # Search window size
    maxLevel=2,            # Number of pyramid levels (for large motions)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Random colors for drawing tracks
colors = np.random.randint(0, 255, (200, 3))

# Read first frame and find corners
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create mask for drawing flow tracks
track_mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow (track points from old_gray to frame_gray)
    p1, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )
    
    # status = 1 means the point was found in the new frame
    # Select only points that were successfully tracked
    if p1 is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]
    
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        
        # Draw line from old position to new position
        track_mask = cv2.line(track_mask, (a, b), (c, d),
                              colors[i % 200].tolist(), 2)
        # Draw circle at current position
        frame = cv2.circle(frame, (a, b), 4,
                          colors[i % 200].tolist(), -1)
    
    # Overlay tracks on frame
    output = cv2.add(frame, track_mask)
    cv2.imshow('Sparse Optical Flow', output)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
```

**Key function: `cv2.calcOpticalFlowPyrLK()`**

| Parameter | Meaning |
|-----------|---------|
| `prevImg` | Previous frame (grayscale) |
| `nextImg` | Current frame (grayscale) |
| `prevPts` | Points to track (from previous frame) |
| `nextPts` | Where the points moved to (output) — or `None` to let OpenCV compute |
| `winSize` | Size of the search window around each point |
| `maxLevel` | Number of pyramid levels (higher = handles bigger motions) |

**Returns:**
- `p1`: New positions of the tracked points
- `status`: Array of 1s and 0s (1 = point found, 0 = lost)
- `error`: Tracking error for each point

### 4.4 Dense Optical Flow: Farneback

Unlike Lucas-Kanade which only tracks selected points, Farneback computes the optical flow for **every single pixel** in the image. This gives you a complete picture of all motion.

The result is typically visualized using the **HSV color space**: the **hue** represents the direction of motion and the **value** (brightness) represents the speed.

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('airport_video.mp4')

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create HSV image for visualization
hsv_mask = np.zeros_like(old_frame)
hsv_mask[..., 1] = 255  # Full saturation

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray,
        None,            # Output (None = create new)
        pyr_scale=0.5,   # Pyramid scale (0.5 = each layer is half the size)
        levels=3,        # Number of pyramid levels
        winsize=15,       # Averaging window size
        iterations=3,    # Iterations at each pyramid level
        poly_n=5,        # Size of pixel neighborhood
        poly_sigma=1.2,  # Gaussian standard deviation
        flags=0
    )
    
    # flow shape: (height, width, 2) — dx and dy for every pixel
    
    # Convert flow to polar coordinates (magnitude and angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Map angle to hue (0-180 in OpenCV HSV)
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    
    # Map magnitude to value (brightness)
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to BGR for display
    flow_vis = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('Dense Optical Flow', flow_vis)
    cv2.imshow('Original', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
```

**Reading the flow visualization:**
- **Color = direction** of motion (red = right, green = down, blue = left, yellow = up)
- **Brightness = speed** (brighter = faster motion, dark = slow or no motion)
- **Black areas = no motion** (background)

### 4.5 Sparse vs Dense — When to Use Which

| Criterion | Sparse (Lucas-Kanade) | Dense (Farneback) |
|-----------|----------------------|-------------------|
| **Speed** | Fast (~100+ FPS) | Slow (~10-30 FPS) |
| **Coverage** | Only tracked points | Every pixel |
| **Best for** | Tracking specific objects | Complete motion analysis |
| **Handles large motion** | Yes (with pyramids) | Yes (with pyramids) |
| **Noise sensitivity** | Lower (fewer points) | Higher (all pixels) |
| **Our project** | Tracking aircraft parts | Overall activity heatmap |

**Practical tip:** In most real projects, start with sparse optical flow because it's faster and gives you actionable information (where specific things moved). Use dense flow when you need to understand the complete motion pattern of a scene, such as generating a motion heatmap.

### 4.6 Computing Motion Statistics

Optical flow vectors give you quantitative data about motion that you can use for analysis:

```python
# After computing dense flow:
flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Compute magnitude at every pixel
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# Average motion across the entire frame
avg_motion = np.mean(magnitude)

# Maximum motion (fastest moving pixel)
max_motion = np.max(magnitude)

# Percentage of pixels with significant motion
motion_threshold = 2.0  # pixels of movement
motion_percent = np.sum(magnitude > motion_threshold) / magnitude.size * 100

print(f"Average motion: {avg_motion:.2f} px")
print(f"Max motion: {max_motion:.2f} px")
print(f"Pixels in motion: {motion_percent:.1f}%")
```

This kind of quantitative motion analysis is exactly what we need for the airplane turnaround project — we can plot motion levels over time to identify when activity starts, peaks, and stops.

---

## 5. Object Tracking — Following Specific Targets

### 5.1 Detection vs Tracking

Up to now, we've been **detecting** motion — finding where movement occurs in each frame independently. **Tracking** is different: it follows a **specific identified object** from frame to frame, maintaining its identity over time.

| Aspect | Detection | Tracking |
|--------|-----------|----------|
| **What it does** | Finds all objects in each frame | Follows one specific object across frames |
| **Identity** | No identity — just "there's an object here" | Maintains identity — "this is object #3" |
| **Speed** | Slower (analyzes full frame each time) | Faster (searches near the last known position) |
| **Handles occlusion** | Can re-detect after occlusion | May lose track if object is hidden |
| **Input needed** | None (automatic) | Initial bounding box (you tell it what to track) |

In practice, you often combine both: use detection to find objects initially, then use tracking to follow them efficiently between detection runs.

### 5.2 OpenCV's Built-in Trackers

OpenCV provides several single-object tracking algorithms through a unified API. You create a tracker, initialize it with a bounding box around the object you want to track, then update it on each new frame.

```python
import cv2

cap = cv2.VideoCapture('airport_video.mp4')

# Read the first frame
ret, frame = cap.read()

# Select a Region of Interest (ROI) — the object to track
# This opens a window where you draw a rectangle around the object
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Create a tracker (choose one)
tracker = cv2.TrackerCSRT_create()     # Most accurate
# tracker = cv2.TrackerKCF_create()    # Good balance
# tracker = cv2.legacy.TrackerMOSSE_create()   # Fastest

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update tracker — it predicts the new position
    success, bbox = tracker.update(frame)
    
    if success:
        # Tracking succeeded — draw the bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Tracking failed — object lost
        cv2.putText(frame, "Lost", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Object Tracking', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 5.3 Tracker Comparison — Choosing the Right One

| Tracker | Speed | Accuracy | Handles Occlusion | Best For |
|---------|-------|----------|-------------------|----------|
| **CSRT** | Slow (~25 FPS) | Highest | Moderate | When accuracy matters most |
| **KCF** | Fast (~80 FPS) | Good | Poor | Real-time tracking with decent quality |
| **MOSSE** | Very Fast (~450 FPS) | Fair | Poor | Ultra-fast tracking, simple objects |
| **MIL** | Medium | Good | Moderate | Objects with appearance changes |
| **MedianFlow** | Fast | Good | Very Poor | Predictable, smooth motion |

**CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability):**
- Uses spatial reliability maps to adjust the filter support to the part of the selected region suitable for tracking
- Operates on HOG (histogram of oriented gradients) features
- Best accuracy among classical trackers
- Our recommendation for the airplane project

**KCF (Kernelized Correlation Filters):**
- Uses circulant matrices for efficient computation
- Exploits the cyclic structure of detection samples
- Good compromise between speed and accuracy

**MOSSE (Minimum Output Sum of Squared Error):**
- Produces a stable correlation filter using a single frame
- Extremely fast but less accurate
- Use when frame rate is critical and objects are simple

### 5.4 Handling Tracker Failure

Trackers can lose their target. A robust system needs failure recovery:

```python
import cv2

cap = cv2.VideoCapture('airport_video.mp4')
ret, frame = cap.read()

bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
cv2.destroyWindow("Select Object")

tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

consecutive_failures = 0
max_failures = 30  # Re-initialize after 30 consecutive failures

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    success, bbox = tracker.update(frame)
    
    if success:
        consecutive_failures = 0
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        consecutive_failures += 1
        
        if consecutive_failures >= max_failures:
            cv2.putText(frame, "LOST - Press 's' to re-select", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Tracking', frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        bbox = cv2.selectROI("Re-select", frame, fromCenter=False)
        cv2.destroyWindow("Re-select")
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        consecutive_failures = 0

cap.release()
cv2.destroyAllWindows()
```

---

## 6. Multi-Object Tracking — Following Many Targets

### 6.1 Why Multi-Object Tracking?

In our airplane turnaround project, we don't just need to track one thing — we need to track the airplane, ground vehicles, jet bridges, fuel trucks, and crew members simultaneously. Single-object trackers can only follow one target. Multi-object tracking (MOT) handles many targets at once.

### 6.2 Simple Multi-Object Tracking with OpenCV

OpenCV doesn't have a built-in multi-object tracker class in recent versions, but you can manage multiple single-object trackers manually:

```python
import cv2

cap = cv2.VideoCapture('airport_video.mp4')
ret, frame = cap.read()

# Create multiple trackers
trackers = []
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

print("Select objects to track. Press ENTER after each, ESC when done.")

while True:
    bbox = cv2.selectROI("Select Objects", frame, fromCenter=False)
    if bbox == (0, 0, 0, 0):
        break  # User pressed ESC
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    trackers.append(tracker)
    print(f"Tracking object #{len(trackers)}")

cv2.destroyWindow("Select Objects")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            color = colors[i % len(colors)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'Object {i+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Multi-Object Tracking', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 6.3 The SORT Algorithm — Automatic Multi-Object Tracking

SORT (Simple Online and Realtime Tracking) is a popular algorithm that combines detection with tracking automatically. Instead of manually selecting objects, SORT uses a detector (like YOLO) to find objects in each frame, then uses the **Kalman filter** and the **Hungarian algorithm** to associate detections across frames.

**How SORT works:**

```
For each frame:
1. Run object detector → get bounding boxes for all objects
2. Predict where existing tracked objects should be (Kalman filter)
3. Match predicted positions to new detections (Hungarian algorithm)
4. Update tracks with matched detections
5. Create new tracks for unmatched detections
6. Delete tracks that haven't been matched for too long
```

**Key components:**

**Kalman Filter:** A mathematical model that predicts where an object will be in the next frame based on its current position and velocity. It handles uncertainty — if a detection is noisy, the Kalman filter smooths the trajectory.

**Hungarian Algorithm:** A combinatorial optimization algorithm that finds the optimal one-to-one assignment between predicted positions and detected positions, minimizing the total distance.

### 6.4 DeepSORT — Adding Appearance Features

DeepSORT improves on SORT by adding a **deep learning appearance model**. While SORT only uses position and size for matching, DeepSORT also uses visual appearance — what the object looks like.

**Why this matters:**
- When two objects cross paths, SORT might swap their identities (ID switch)
- DeepSORT recognizes that the objects look different, preventing the swap
- DeepSORT reduces identity switches by ~45% compared to SORT

**DeepSORT pipeline:**
1. **Detect** objects using YOLO or similar detector
2. **Extract appearance features** using a pre-trained CNN (128-dimensional embedding)
3. **Predict** track positions using Kalman filter
4. **Match** using a combination of:
   - **Mahalanobis distance** (motion-based) — are they in the expected position?
   - **Cosine distance** (appearance-based) — do they look like the same object?
5. **Update** matched tracks, create new ones, delete old ones

```python
# DeepSORT conceptual usage (requires deep-sort-realtime library)
# pip install deep-sort-realtime

from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30, n_init=3)

# For each frame:
# detections = your_detector(frame)  # Get bounding boxes
# Format: [[x1, y1, w, h, confidence], ...]

# tracks = tracker.update_tracks(detections, frame=frame)
# for track in tracks:
#     if track.is_confirmed():
#         track_id = track.track_id
#         bbox = track.to_ltrb()  # left, top, right, bottom
```

### 6.5 MOT Metrics — How Good is Your Tracking?

When evaluating multi-object tracking, several metrics are used:

| Metric | What It Measures | Ideal Value |
|--------|-----------------|-------------|
| **MOTA** (Multiple Object Tracking Accuracy) | Overall accuracy (combines FP, FN, ID switches) | 100% |
| **IDF1** (ID F1 Score) | How well identities are preserved | 100% |
| **ID Switches** | Times an object's ID changes incorrectly | 0 |
| **FP** (False Positives) | Tracked objects that don't exist | 0 |
| **FN** (False Negatives) | Real objects that weren't tracked | 0 |
| **MT** (Mostly Tracked) | % of objects tracked for >80% of their lifetime | 100% |
| **ML** (Mostly Lost) | % of objects tracked for <20% of their lifetime | 0% |

---

## 7. Building a Motion Analysis System — Putting It All Together

### 7.1 Complete Video Analysis Pipeline

Here's a comprehensive pipeline that combines frame differencing, background subtraction, and contour analysis into a working motion detection system:

```python
import cv2
import numpy as np
from collections import deque

class MotionAnalyzer:
    def __init__(self, video_path, min_area=1500):
        self.cap = cv2.VideoCapture(video_path)
        self.min_area = min_area
        
        # Background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Morphological kernels
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Motion history
        self.motion_history = deque(maxlen=300)
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0
    
    def process_frame(self, frame):
        self.frame_count += 1
        timestamp = self.frame_count / self.fps
        
        # Step 1: Background subtraction
        fg_mask = self.bg_sub.apply(frame)
        
        # Step 2: Remove shadows (keep only definite foreground)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Step 3: Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Step 5: Filter and analyze
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'area': area,
                    'timestamp': timestamp
                })
        
        # Step 6: Record motion level
        motion_level = np.sum(fg_mask > 0) / fg_mask.size * 100
        self.motion_history.append({
            'frame': self.frame_count,
            'time': timestamp,
            'motion_percent': motion_level,
            'num_objects': len(detections)
        })
        
        return detections, fg_mask, motion_level
    
    def draw_results(self, frame, detections, motion_level):
        output = frame.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = det['centroid']
            
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            
            label = f"Area: {det['area']}"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # HUD overlay
        info = f"Frame: {self.frame_count} | Objects: {len(detections)} | Motion: {motion_level:.1f}%"
        cv2.putText(output, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            detections, mask, motion = self.process_frame(frame)
            output = self.draw_results(frame, detections, motion)
            
            cv2.imshow('Motion Analysis', output)
            cv2.imshow('Foreground Mask', mask)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        return list(self.motion_history)


# Usage
analyzer = MotionAnalyzer('airport_video.mp4', min_area=2000)
history = analyzer.run()

# Analyze motion over time
for entry in history[-10:]:
    print(f"t={entry['time']:.1f}s: {entry['motion_percent']:.1f}% motion, "
          f"{entry['num_objects']} objects")
```

### 7.2 Motion Timeline Analysis

For the airplane turnaround project, plotting motion levels over time reveals the phases of aircraft servicing:

```python
import matplotlib.pyplot as plt

def plot_motion_timeline(motion_history):
    times = [h['time'] for h in motion_history]
    motion = [h['motion_percent'] for h in motion_history]
    objects = [h['num_objects'] for h in motion_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Motion percentage over time
    ax1.fill_between(times, motion, alpha=0.3, color='cyan')
    ax1.plot(times, motion, color='cyan', linewidth=0.5)
    ax1.set_ylabel('Motion %')
    ax1.set_title('Motion Activity Over Time')
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Activity threshold')
    ax1.legend()
    
    # Number of detected objects over time
    ax2.bar(times, objects, width=1/30, color='orange', alpha=0.7)
    ax2.set_ylabel('Objects Detected')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Object Count Over Time')
    
    plt.tight_layout()
    plt.savefig('motion_timeline.png', dpi=150)
    plt.show()
```

This timeline would show patterns like:
- **Low motion** → airplane parked, no ground activity
- **Rising motion** → ground vehicles approaching, jet bridge extending
- **High sustained motion** → active servicing (fueling, baggage, catering)
- **Declining motion** → servicing complete, vehicles departing
- **Brief motion spike** → jet bridge retracting, pushback beginning

---

## 8. Advanced Topics — Region of Interest & Activity Zones

### 8.1 Defining Regions of Interest (ROI)

In the airplane video, not every pixel matters equally. The runway in the distance, the sky, or the terminal building are irrelevant. We only care about the tarmac area where the airplane is parked and serviced.

Defining ROIs focuses your analysis and dramatically reduces false positives:

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('airport_video.mp4')
ret, frame = cap.read()

# Define a polygon ROI (the tarmac area)
# These points would be determined by looking at the video
roi_points = np.array([
    [100, 400], [500, 300], [900, 300],
    [1100, 400], [1100, 700], [100, 700]
], dtype=np.int32)

# Create a mask from the polygon
roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(roi_mask, [roi_points], 255)

bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fg_mask = bg_sub.apply(frame)
    
    # Apply ROI mask — only keep motion inside the ROI
    fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
    
    # Continue with contour detection as before...
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    # Draw the ROI boundary on the frame
    cv2.polylines(frame, [roi_points], True, (0, 255, 255), 2)
    
    cv2.imshow('ROI Analysis', frame)
    cv2.imshow('Masked Foreground', fg_mask)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 8.2 Activity Zone Detection

You can divide the tarmac into zones and monitor activity in each zone separately:

```python
zones = {
    'jet_bridge': np.array([[200,300],[400,300],[400,500],[200,500]]),
    'fuel_area':  np.array([[500,350],[700,350],[700,550],[500,550]]),
    'baggage':    np.array([[750,400],[950,400],[950,600],[750,600]]),
    'pushback':   np.array([[100,350],[200,350],[200,600],[100,600]]),
}

def analyze_zones(fg_mask, zones):
    results = {}
    for name, polygon in zones.items():
        zone_mask = np.zeros_like(fg_mask)
        cv2.fillPoly(zone_mask, [polygon], 255)
        
        zone_fg = cv2.bitwise_and(fg_mask, zone_mask)
        zone_pixels = np.sum(zone_mask > 0)
        active_pixels = np.sum(zone_fg > 0)
        
        activity = (active_pixels / zone_pixels * 100) if zone_pixels > 0 else 0
        results[name] = {
            'activity_percent': activity,
            'is_active': activity > 5.0
        }
    
    return results
```

### 8.3 Event Detection

By monitoring zone activity over time, you can detect specific turnaround events:

```python
class EventDetector:
    def __init__(self, threshold=5.0, min_duration_frames=30):
        self.threshold = threshold
        self.min_duration = min_duration_frames
        self.zone_states = {}  # zone_name -> {'active_since': frame, 'is_active': bool}
        self.events = []
    
    def update(self, zone_results, frame_num, fps):
        for zone_name, data in zone_results.items():
            if zone_name not in self.zone_states:
                self.zone_states[zone_name] = {'active_since': None, 'is_active': False}
            
            state = self.zone_states[zone_name]
            
            if data['is_active'] and not state['is_active']:
                # Activity just started
                state['active_since'] = frame_num
                state['is_active'] = True
            
            elif not data['is_active'] and state['is_active']:
                # Activity just stopped
                duration_frames = frame_num - state['active_since']
                
                if duration_frames >= self.min_duration:
                    duration_sec = duration_frames / fps
                    self.events.append({
                        'zone': zone_name,
                        'start_frame': state['active_since'],
                        'end_frame': frame_num,
                        'start_time': state['active_since'] / fps,
                        'end_time': frame_num / fps,
                        'duration': duration_sec
                    })
                
                state['is_active'] = False
                state['active_since'] = None
        
        return self.events
```

---

## 9. Connection to the Airplane Turnaround Project

Every technique from today maps directly to the turnaround time calculator:

| Technique | Project Application |
|-----------|-------------------|
| **Video I/O** | Reading the surveillance video frame by frame, saving annotated output |
| **Frame differencing** | Quick initial check for when activity begins after airplane parks |
| **Background subtraction** | Detecting all ground equipment and personnel as they arrive and depart |
| **Optical flow (sparse)** | Tracking the jet bridge as it extends to the aircraft door |
| **Optical flow (dense)** | Creating an overall activity heatmap of the tarmac |
| **Single-object tracking** | Following specific vehicles (fuel truck, catering truck, baggage cart) |
| **Multi-object tracking** | Tracking all objects simultaneously with unique IDs |
| **ROI / Zone analysis** | Monitoring activity in specific areas (fuel zone, baggage zone, pushback zone) |
| **Event detection** | Identifying when each phase of servicing starts and ends |
| **Motion timeline** | Plotting activity levels to determine total turnaround time |

**Turnaround time = time between airplane arrival (high motion spike) and departure (pushback motion)**

The techniques we learned across Days 1-3 give you a complete classical computer vision toolkit:
- Day 1: Read and preprocess images
- Day 2: Extract structure (edges, contours, features) from each frame
- Day 3: Analyze sequences of frames (video) to detect and track motion

Tomorrow (Day 4) we'll add deep learning — using YOLO to automatically detect and classify the specific objects (airplane, fuel truck, jet bridge) rather than just detecting generic motion blobs.

---

## 10. Key Takeaways from Day 3

1. **Video is just a sequence of images.** Everything from Day 1 and Day 2 applies — you just run it inside a `while cap.read()` loop.

2. **Frame differencing is the simplest motion detection.** Subtract consecutive frames, threshold, find contours. Fast but fragile — fails with lighting changes and camera shake.

3. **Background subtraction builds a statistical model.** MOG2 and KNN learn what the background looks like and detect anything new as foreground. Much more robust than frame differencing.

4. **Optical flow tells you direction and speed.** Sparse (Lucas-Kanade) tracks specific points fast. Dense (Farneback) computes motion for every pixel but is slower.

5. **Tracking maintains object identity.** CSRT is most accurate, KCF balances speed and accuracy, MOSSE is fastest. Choose based on your FPS requirements.

6. **Multi-object tracking combines detection and tracking.** SORT uses Kalman filter + Hungarian algorithm. DeepSORT adds appearance features to prevent ID switches.

7. **Zones and events give you actionable data.** By monitoring activity in specific regions over time, you can detect when each phase of the turnaround process starts and ends.

---

## Preparation for Day 4

**Practice:** Run the MotionAnalyzer class from Section 7 on the airplane video. Try adjusting `min_area` and the `varThreshold` to see how they affect detection quality.

**Think ahead:** Tomorrow we add intelligence. Instead of detecting generic "motion blobs," we'll use YOLO to say "this is a fuel truck" and "that is a jet bridge." Start thinking about what specific objects are visible in the airplane video and how you'd want to classify them.

**Install:** Make sure you have the `ultralytics` package installed for tomorrow:
```bash
pip install ultralytics
```

---

**End of Day 3 Lecture Notes**
