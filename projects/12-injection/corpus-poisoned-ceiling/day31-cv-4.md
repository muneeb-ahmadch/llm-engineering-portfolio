# Day 4: Deep Learning for Computer Vision — YOLO, Detection & The Turnaround Pipeline

## Quick Recap from Days 1–3

On **Day 1**, we learned how digital images work — pixels, color spaces, filtering, and basic OpenCV operations. On **Day 2**, we extracted structure from images — edges with Canny, shapes with contours, and distinctive keypoints with SIFT and ORB. On **Day 3**, we made everything dynamic — processing video frame by frame, detecting motion with background subtraction, measuring movement with optical flow, and tracking objects with CSRT and DeepSORT.

All of that was **classical computer vision** — hand-crafted algorithms that follow explicit rules. Today we add **intelligence**. Instead of detecting generic "motion blobs," we'll use deep learning to say "this is a fuel truck" and "that is a jet bridge." Instead of tracking anonymous contours, we'll track identified, classified objects with persistent labels.

**Today's learning path:**

```
Why Deep Learning? → Object Detection Fundamentals → YOLO Architecture → Using YOLO in Python →
Custom Training → Building the Turnaround Pipeline → Putting Everything Together
```

By the end of today, you'll have a complete pipeline that combines classical CV (Days 1-3) with deep learning (Day 4) to analyze the airplane turnaround video.

---

## 1. Why Deep Learning for Computer Vision?

### 1.1 The Limitations of Classical CV

Over the past three days, we built a powerful toolkit: edge detection, contour analysis, background subtraction, optical flow, tracking. But there's a fundamental limitation — **classical CV can detect motion and shapes, but it can't understand what things are**.

Consider our airplane turnaround video:
- Background subtraction detects a "moving blob" — but is it a fuel truck or a baggage cart?
- Contour analysis finds a "large rectangle" — but is it the airplane or the terminal building?
- Optical flow shows "something moved left" — but was it a ground vehicle departing or arriving?

Classical CV answers "where is motion?" but not "what is this object?" For our project, we need both.

### 1.2 How Deep Learning Sees

Deep learning approaches this differently. Instead of hand-crafting rules like "an edge is where the gradient exceeds a threshold," deep learning **learns from examples**. You show it thousands of labeled images — "this is a fuel truck," "this is a jet bridge" — and it learns the visual patterns that distinguish each class.

**Convolutional Neural Networks (CNNs)** are the foundation of deep learning for vision:

```
Input Image → [Conv layers extract features] → [Fully connected layers classify] → Output
                    ↓                                        ↓
              Low-level: edges, textures         "fuel truck" (92% confidence)
              Mid-level: shapes, patterns
              High-level: parts, objects
```

The key insight: early layers learn simple features (edges, colors — similar to what we did on Day 2!), middle layers combine them into patterns (wheels, windows), and deep layers recognize whole objects (fuel truck, airplane).

### 1.3 Classical CV + Deep Learning: The Best of Both Worlds

Modern CV pipelines don't choose between classical and deep learning — they use both:

| Task | Best Approach |
|------|--------------|
| **Image preprocessing** | Classical CV (grayscale, blur, resize) — Day 1 |
| **Edge/contour analysis** | Classical CV (Canny, findContours) — Day 2 |
| **Background subtraction** | Classical CV (MOG2) — Day 3 |
| **Object classification** | Deep Learning (CNN, what is this?) — Day 4 |
| **Object detection** | Deep Learning (YOLO, what is this and where?) — Day 4 |
| **Object tracking** | Hybrid (YOLO + DeepSORT, track identified objects) — Day 3+4 |

Our turnaround pipeline will use classical CV for video processing, motion detection, and zone analysis, and deep learning for identifying specific ground equipment.

---

## 2. Object Detection — What and Where

### 2.1 Three Levels of Visual Understanding

Deep learning for vision operates at three increasing levels of detail:

**Image Classification:** "This image contains a fuel truck." One label for the entire image. No information about where.

**Object Detection:** "There is a fuel truck at position (x=100, y=200, w=150, h=100)." Labels AND bounding boxes. Tells you what objects are present and where each one is located.

**Instance Segmentation:** "Every pixel belonging to the fuel truck is labeled." Pixel-perfect boundaries for each object.

For our project, **object detection** is the sweet spot — we need to know both what objects are present and where they are, but pixel-perfect segmentation isn't necessary.

### 2.2 How Object Detection Works

An object detector takes an image and outputs a list of detections, where each detection contains:

```python
detection = {
    'class': 'fuel_truck',      # What is it?
    'confidence': 0.92,          # How sure are we? (0-1)
    'bbox': [x1, y1, x2, y2]   # Where is it? (bounding box)
}
```

The detector must solve two problems simultaneously:
1. **Classification:** What type of object is this? (fuel truck, baggage cart, airplane...)
2. **Localization:** Where exactly is it? (bounding box coordinates)

### 2.3 Two-Stage vs One-Stage Detectors

**Two-stage detectors (Faster R-CNN):**
1. Stage 1: Propose regions that might contain objects (Region Proposal Network)
2. Stage 2: Classify and refine each proposed region
- Pros: Higher accuracy, especially for small objects
- Cons: Slower (5-10 FPS typically)
- Use when: Accuracy is more important than speed

**One-stage detectors (YOLO, SSD):**
1. Single pass: Predict classes and boxes simultaneously across the entire image
- Pros: Much faster (30-150+ FPS), real-time capable
- Cons: Slightly lower accuracy on small objects
- Use when: Speed matters (real-time video, our project!)

### 2.4 Key Detection Metrics

| Metric | What It Measures | Ideal |
|--------|-----------------|-------|
| **Precision** | Of all detections, how many were correct? | 100% |
| **Recall** | Of all real objects, how many were detected? | 100% |
| **mAP (Mean Average Precision)** | Overall detection quality across all classes | 100% |
| **IoU (Intersection over Union)** | How well does the predicted box overlap the real box? | 1.0 |
| **FPS** | How many frames per second can be processed | Higher = better |

**IoU explained:**
```
IoU = Area of Overlap / Area of Union

IoU = 0.0  →  No overlap (completely wrong)
IoU = 0.5  →  50% overlap (minimum acceptable)
IoU = 1.0  →  Perfect overlap (exact match)
```

Typically, a detection is considered "correct" if IoU ≥ 0.5 and the class label is right.

---

## 3. YOLO — You Only Look Once

### 3.1 What Makes YOLO Special

YOLO (You Only Look Once) revolutionized object detection when it was introduced in 2015. The key innovation: instead of scanning the image with a sliding window or proposing regions, YOLO processes the **entire image in a single forward pass** through the neural network.

**How YOLO works:**
1. Divide the image into an S × S grid (e.g., 13 × 13)
2. Each grid cell predicts B bounding boxes + confidence scores
3. Each grid cell also predicts C class probabilities
4. Non-Maximum Suppression (NMS) removes duplicate detections

This "look at everything at once" approach is why YOLO is so fast — one forward pass gives you all detections.

### 3.2 YOLO Evolution

| Version | Year | Key Innovation | Speed | Accuracy |
|---------|------|---------------|-------|----------|
| YOLOv1 | 2015 | Single-pass detection concept | 45 FPS | Low |
| YOLOv2 | 2016 | Batch norm, anchor boxes, multi-scale | 67 FPS | Better |
| YOLOv3 | 2018 | Feature Pyramid Network, 3 scales | 30 FPS | Good |
| YOLOv4 | 2020 | CSPDarknet, PANet, bag of freebies | 65 FPS | Very Good |
| YOLOv5 | 2020 | PyTorch-native, easy to use | 140 FPS | Very Good |
| YOLOv8 | 2023 | Anchor-free, Ultralytics framework | 160 FPS | Excellent |
| YOLO11 | 2024 | C3k2 blocks, C2PSA attention, 22% fewer params | 180+ FPS | State of Art |

**We'll use YOLO11** (latest from Ultralytics) — it's the fastest, most accurate, and easiest to use.

### 3.3 YOLO Model Sizes

YOLO comes in multiple sizes — trade off speed for accuracy:

| Model | Parameters | mAP (COCO) | Speed (GPU) | Best For |
|-------|-----------|------------|-------------|----------|
| YOLO11n (Nano) | 2.6M | 39.5% | 1.5ms | Edge devices, mobile |
| YOLO11s (Small) | 9.4M | 47.0% | 2.5ms | Embedded, real-time |
| YOLO11m (Medium) | 20.1M | 51.5% | 4.7ms | Balanced accuracy/speed |
| YOLO11l (Large) | 25.3M | 53.4% | 6.2ms | High accuracy |
| YOLO11x (XLarge) | 56.9M | 54.7% | 11.3ms | Maximum accuracy |

For our project, **YOLO11m or YOLO11s** is recommended — good accuracy with real-time speed on a regular GPU.

### 3.4 What YOLO Can Detect Out of the Box

Pre-trained YOLO models are trained on the **COCO dataset** (Common Objects in Context) which includes 80 classes:

- **Vehicles:** car, bus, truck, motorcycle, bicycle, airplane, boat, train
- **People:** person
- **Animals:** dog, cat, horse, bird, cow, sheep, elephant, bear, zebra, giraffe
- **Objects:** chair, couch, bed, table, TV, laptop, cell phone, book, clock, etc.

For our airport project, the pre-trained model can already detect: **airplane, truck, car, person, bus**. For specialized equipment (fuel truck, jet bridge, baggage cart), we may need custom training.

---

## 4. Using YOLO in Python — Hands-On

### 4.1 Installation

```bash
pip install ultralytics
```

That's it — one package gives you everything: model loading, inference, training, export.

### 4.2 Basic Object Detection on an Image

```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolo11m.pt')  # Downloads automatically on first use

# Run detection on an image
results = model('airport_frame.jpg')

# Results is a list (one per image)
for result in results:
    boxes = result.boxes  # All detected bounding boxes
    
    for box in boxes:
        # Bounding box coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Confidence score (0-1)
        confidence = box.conf[0].item()
        
        # Class ID and name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        print(f"{class_name}: {confidence:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Save annotated image
result.save('output_detection.jpg')
```

### 4.3 Detection on Video

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolo11m.pt')

cap = cv2.VideoCapture('Airplane_Video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('detected_output.mp4', fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    timestamp = frame_count / fps
    
    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)
    
    # Draw results on the frame
    annotated_frame = results[0].plot()
    
    # Extract detection info
    for box in results[0].boxes:
        class_name = model.names[int(box.cls[0])]
        conf = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        print(f"[{timestamp:.1f}s] {class_name}: {conf:.2f}")
    
    out.write(annotated_frame)
    cv2.imshow('YOLO Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

### 4.4 Filtering Detections

In our airport video, we don't care about detecting cups or books. Filter to relevant classes only:

```python
# COCO class IDs for airport-relevant objects
RELEVANT_CLASSES = {
    0: 'person',
    2: 'car',
    4: 'airplane',
    5: 'bus',
    7: 'truck',
}

results = model(frame, conf=0.5, classes=list(RELEVANT_CLASSES.keys()))
```

### 4.5 Key Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `conf` | 0.25 | Minimum confidence threshold (higher = fewer but more certain detections) |
| `iou` | 0.7 | IoU threshold for NMS (higher = stricter duplicate removal) |
| `classes` | None | Filter to specific class IDs only |
| `imgsz` | 640 | Input image size (larger = slower but more accurate) |
| `device` | auto | `'cpu'`, `'cuda:0'`, or `'mps'` for Apple Silicon |
| `verbose` | True | Print results to console |

---

## 5. Custom Training — Teaching YOLO New Objects

### 5.1 Why Custom Training?

The pre-trained YOLO model knows 80 COCO classes. But for the airport turnaround project, we may need to detect specific objects that COCO doesn't include:

| Object | In COCO? | Notes |
|--------|----------|-------|
| Airplane | ✓ | Detectable with pre-trained model |
| Truck (generic) | ✓ | Detectable, but won't distinguish fuel vs baggage truck |
| Person | ✓ | Ground crew detection |
| Jet bridge | ✗ | Needs custom training |
| Fuel truck | ✗ | Needs custom training or labeled as "truck" |
| Baggage cart | ✗ | Needs custom training |
| Catering vehicle | ✗ | Needs custom training |
| Pushback tug | ✗ | Needs custom training |

### 5.2 Custom Training Pipeline

```
1. Collect Images  →  2. Annotate  →  3. Organize Dataset  →  4. Train  →  5. Evaluate  →  6. Deploy
```

**Step 1: Collect Images**
- Extract frames from the airport video at intervals
- Find additional airport ground operations images online
- Aim for 100+ images per class (more is better)

```python
# Extract frames from video
cap = cv2.VideoCapture('Airplane_Video.mp4')
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_num % 30 == 0:  # Every 30th frame
        cv2.imwrite(f'frames/frame_{frame_num:05d}.jpg', frame)
    frame_num += 1
cap.release()
```

**Step 2: Annotate with Roboflow**
- Upload images to [Roboflow](https://roboflow.com) (free tier available)
- Draw bounding boxes around each object
- Label each box with its class name
- Roboflow exports in YOLO format automatically

**Step 3: YOLO Annotation Format**

Each image gets a corresponding `.txt` file with one line per object:

```
# Format: class_id  center_x  center_y  width  height  (all normalized 0-1)
0  0.5  0.4  0.3  0.2    ← class 0 (airplane) at center (0.5, 0.4), size 30% × 20%
1  0.2  0.7  0.1  0.15   ← class 1 (fuel_truck) at center (0.2, 0.7), size 10% × 15%
```

**Step 4: Dataset Structure**

```
dataset/
├── data.yaml           # Dataset configuration
├── train/
│   ├── images/        # Training images
│   └── labels/        # Corresponding label files
├── val/
│   ├── images/        # Validation images
│   └── labels/        # Corresponding label files
└── test/
    ├── images/        # Test images (optional)
    └── labels/
```

**data.yaml:**
```yaml
train: train/images
val: val/images
test: test/images

nc: 6  # Number of classes
names: ['airplane', 'fuel_truck', 'baggage_cart', 'catering', 'jet_bridge', 'pushback_tug']
```

**Step 5: Train**

```python
from ultralytics import YOLO

# Start from a pre-trained model (transfer learning!)
model = YOLO('yolo11m.pt')

# Train on custom dataset
results = model.train(
    data='dataset/data.yaml',
    epochs=100,           # Training iterations
    imgsz=640,           # Image size
    batch=16,            # Batch size (reduce if out of memory)
    patience=20,         # Early stopping patience
    save=True,           # Save best model
    project='airport',   # Output directory
    name='turnaround_v1' # Experiment name
)
```

**Step 6: Evaluate**

```python
# Evaluate on validation set
metrics = model.val()

print(f"mAP50: {metrics.box.map50:.3f}")     # mAP at IoU 0.50
print(f"mAP50-95: {metrics.box.map:.3f}")     # mAP at IoU 0.50:0.95
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
```

**Step 7: Use Your Custom Model**

```python
# Load your trained model
model = YOLO('airport/turnaround_v1/weights/best.pt')

# Run inference — same API as before!
results = model('airport_frame.jpg')
```

### 5.3 Transfer Learning — Why Start From Pre-Trained

Training from scratch requires millions of images and weeks of GPU time. **Transfer learning** starts from a model that already understands visual features (edges, textures, shapes) and only needs to learn the new classes.

This is why we start from `yolo11m.pt` — it already knows what wheels, windows, and vehicle shapes look like. We just teach it to distinguish between a fuel truck and a baggage cart.

---

## 6. Building the Turnaround Pipeline — Combining Everything

### 6.1 Architecture Overview

Our complete turnaround time calculator combines classical CV (Days 1-3) with deep learning (Day 4):

```
Video Input
    │
    ├──→ YOLO Detection (every N frames)
    │        → Detect: airplane, trucks, equipment, people
    │        → Output: class labels + bounding boxes + confidence
    │
    ├──→ Background Subtraction (every frame)
    │        → Foreground mask
    │        → Motion detection
    │
    ├──→ Zone Analysis
    │        → Map detections to tarmac zones
    │        → Calculate zone activity levels
    │
    ├──→ Object Tracking (DeepSORT)
    │        → Persistent IDs for each detected object
    │        → Track trajectories over time
    │
    └──→ Event Detection
             → Log turnaround events with timestamps
             → Calculate total turnaround time
```

### 6.2 Complete Pipeline Code

```python
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import timedelta

class TurnaroundAnalyzer:
    def __init__(self, video_path, model_path='yolo11m.pt'):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # YOLO model
        self.model = YOLO(model_path)
        
        # Background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Tracking state
        self.frame_count = 0
        self.events = []
        self.motion_history = []
        self.detection_history = defaultdict(list)
        
        # Event detection state
        self.airplane_detected = False
        self.airplane_first_seen = None
        self.last_activity_time = None
        self.activity_started = False
    
    def get_timestamp(self):
        seconds = self.frame_count / self.fps
        return str(timedelta(seconds=int(seconds)))
    
    def analyze_frame(self, frame):
        self.frame_count += 1
        timestamp = self.frame_count / self.fps
        
        # ─── YOLO DETECTION (every 5th frame for speed) ───
        detections = []
        if self.frame_count % 5 == 0:
            results = self.model(frame, conf=0.4, verbose=False,
                                classes=[0, 2, 4, 5, 7])
            
            for box in results[0].boxes:
                det = {
                    'class': self.model.names[int(box.cls[0])],
                    'confidence': box.conf[0].item(),
                    'bbox': box.xyxy[0].tolist(),
                    'timestamp': timestamp
                }
                detections.append(det)
                self.detection_history[det['class']].append(timestamp)
        
        # ─── BACKGROUND SUBTRACTION ───
        fg_mask = self.bg_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        motion_level = np.sum(fg_mask > 0) / fg_mask.size * 100
        self.motion_history.append({
            'frame': self.frame_count,
            'time': timestamp,
            'motion': motion_level,
            'detections': len(detections)
        })
        
        # ─── EVENT DETECTION ───
        self.detect_events(detections, motion_level, timestamp)
        
        return detections, fg_mask, motion_level
    
    def detect_events(self, detections, motion_level, timestamp):
        # Check for airplane
        airplane_dets = [d for d in detections if d['class'] == 'airplane']
        
        if airplane_dets and not self.airplane_detected:
            self.airplane_detected = True
            self.airplane_first_seen = timestamp
            self.events.append({
                'time': timestamp,
                'event': 'Aircraft detected at gate',
                'type': 'arrival'
            })
        
        # Check for ground equipment
        equipment = [d for d in detections 
                    if d['class'] in ['truck', 'car', 'bus', 'person']]
        
        if equipment and self.airplane_detected:
            if not self.activity_started:
                self.activity_started = True
                self.events.append({
                    'time': timestamp,
                    'event': f'Ground activity started ({len(equipment)} objects)',
                    'type': 'activity_start'
                })
            self.last_activity_time = timestamp
        
        # Check for high motion (potential pushback)
        if motion_level > 15 and self.airplane_detected and self.activity_started:
            if timestamp - self.last_activity_time > 10:
                self.events.append({
                    'time': timestamp,
                    'event': 'Possible pushback / departure',
                    'type': 'departure'
                })
    
    def draw_results(self, frame, detections, motion_level):
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            
            color = (0, 255, 0)
            if det['class'] == 'airplane':
                color = (255, 100, 0)
            elif det['class'] == 'truck':
                color = (0, 200, 255)
            elif det['class'] == 'person':
                color = (255, 0, 255)
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # HUD overlay
        info = (f"Frame: {self.frame_count}/{self.total_frames} | "
                f"Time: {self.get_timestamp()} | "
                f"Detections: {len(detections)} | "
                f"Motion: {motion_level:.1f}%")
        cv2.putText(output, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output
    
    def generate_report(self):
        report = []
        report.append("=" * 50)
        report.append("  AIRCRAFT TURNAROUND TIME REPORT")
        report.append("=" * 50)
        report.append("")
        report.append(f"Video Duration: {self.total_frames / self.fps:.0f} seconds")
        report.append(f"Total Frames Processed: {self.frame_count}")
        report.append(f"Detection Model: YOLO11")
        report.append("")
        report.append("Key Events Detected:")
        
        for event in self.events:
            time_str = str(timedelta(seconds=int(event['time'])))
            report.append(f"  [{time_str}]  {event['event']}")
        
        # Calculate turnaround time
        arrivals = [e for e in self.events if e['type'] == 'arrival']
        departures = [e for e in self.events if e['type'] == 'departure']
        
        if arrivals and departures:
            tat = departures[-1]['time'] - arrivals[0]['time']
            report.append("")
            report.append("-" * 50)
            report.append(f"  TURNAROUND TIME: {str(timedelta(seconds=int(tat)))}")
            report.append("-" * 50)
        
        # Detection summary
        report.append("")
        report.append("Detection Summary:")
        for cls, times in self.detection_history.items():
            report.append(f"  {cls}: detected in {len(times)} frames")
        
        return "\n".join(report)
    
    def run(self, display=True):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            detections, mask, motion = self.analyze_frame(frame)
            
            if display:
                output = self.draw_results(frame, detections, motion)
                cv2.imshow('Turnaround Analysis', output)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        report = self.generate_report()
        print(report)
        
        # Save report
        with open('turnaround_report.txt', 'w') as f:
            f.write(report)
        
        return report


# ──── Run the pipeline ────
analyzer = TurnaroundAnalyzer('Airplane_Video.mp4')
report = analyzer.run()
```

### 6.3 Adding DeepSORT for Persistent Tracking

To track identified objects with persistent IDs across frames, integrate DeepSORT:

```python
# pip install deep-sort-realtime

from deep_sort_realtime.deepsort_tracker import DeepSort

class TurnaroundAnalyzerWithTracking(TurnaroundAnalyzer):
    def __init__(self, video_path, model_path='yolo11m.pt'):
        super().__init__(video_path, model_path)
        self.tracker = DeepSort(max_age=30, n_init=3)
    
    def analyze_frame(self, frame):
        self.frame_count += 1
        timestamp = self.frame_count / self.fps
        
        # YOLO Detection
        results = self.model(frame, conf=0.4, verbose=False,
                            classes=[0, 2, 4, 5, 7])
        
        # Format detections for DeepSORT
        detections_for_tracker = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            detections_for_tracker.append(
                ([x1, y1, x2-x1, y2-y1], conf, self.model.names[cls])
            )
        
        # Update tracker
        tracks = self.tracker.update_tracks(
            detections_for_tracker, frame=frame
        )
        
        # Process confirmed tracks
        tracked_objects = []
        for track in tracks:
            if track.is_confirmed():
                tracked_objects.append({
                    'track_id': track.track_id,
                    'class': track.det_class,
                    'bbox': track.to_ltrb(),
                    'timestamp': timestamp
                })
        
        # Background subtraction
        fg_mask = self.bg_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        motion_level = np.sum(fg_mask > 0) / fg_mask.size * 100
        
        return tracked_objects, fg_mask, motion_level
```

---

## 7. YOLO Beyond Detection — Other Tasks

### 7.1 Instance Segmentation

YOLO can also perform pixel-level segmentation, giving you the exact outline of each object:

```python
model = YOLO('yolo11m-seg.pt')  # Segmentation model
results = model('airport_frame.jpg')

# Access segmentation masks
for result in results:
    if result.masks is not None:
        masks = result.masks.data  # Binary masks for each detection
```

### 7.2 Pose Estimation

YOLO can detect human body keypoints — useful for detecting ground crew activities:

```python
model = YOLO('yolo11m-pose.pt')  # Pose model
results = model('airport_frame.jpg')

# Access keypoints
for result in results:
    if result.keypoints is not None:
        keypoints = result.keypoints.data  # [N, 17, 3] — 17 body points
```

### 7.3 Image Classification

When you only need to classify the entire image without bounding boxes:

```python
model = YOLO('yolo11m-cls.pt')  # Classification model
results = model('airport_frame.jpg')
```

---

## 8. Performance Optimization

### 8.1 Speed Tips

| Technique | Speedup | How |
|-----------|---------|-----|
| Process every Nth frame | 5-10× | Skip frames: `if frame_count % 5 == 0` |
| Smaller input size | 2-4× | `model(frame, imgsz=320)` instead of 640 |
| Smaller model | 2-3× | Use `yolo11n.pt` instead of `yolo11m.pt` |
| GPU acceleration | 10-50× | `model(frame, device='cuda:0')` |
| Batch processing | 2-3× | Process multiple frames at once |
| Half precision | 1.5-2× | `model(frame, half=True)` on GPU |

### 8.2 Accuracy Tips

| Technique | Improvement | How |
|-----------|------------|-----|
| Higher conf threshold | Fewer false positives | `model(frame, conf=0.6)` |
| Class filtering | Remove irrelevant classes | `model(frame, classes=[0,2,4,7])` |
| Larger input size | Better small object detection | `model(frame, imgsz=1280)` |
| Larger model | Higher mAP | Use `yolo11l.pt` or `yolo11x.pt` |
| Custom training | Domain-specific accuracy | Train on airport images |
| Test-time augmentation | 1-2% mAP boost | `model(frame, augment=True)` |

### 8.3 Exporting Models for Production

YOLO supports exporting to optimized formats for deployment:

```python
model = YOLO('yolo11m.pt')

# Export to ONNX (cross-platform, fast inference)
model.export(format='onnx')

# Export to TensorRT (NVIDIA GPU optimized)
model.export(format='engine')

# Export to CoreML (Apple devices)
model.export(format='coreml')

# Export to TFLite (mobile/edge devices)
model.export(format='tflite')
```

---

## 9. Real-World Airport CV Systems

### 9.1 Industry Applications

Computer vision is already revolutionizing airport ground operations:

**Assaia ApronAI** — Deployed at 20+ airports globally:
- Cameras at each gate automatically detect ground handling events
- Real-time alerts for operational deviations
- Alaska Airlines achieved 17% increase in on-time performance
- Toronto Pearson reduced ground delays by 3.4 minutes per flight

**Schiphol Deep Turnaround** — Amsterdam Airport:
- Camera-based system predicts aircraft departure readiness
- Automatically detects when ground handling starts and stops
- Enables proactive delay prevention

**Key Milestone Nodes (KMN):**
Research shows CV systems can autonomously recognize critical turnaround phases — taxi-in, ground handling activities, taxi-out — with time error margins under 60 seconds, meeting Airport-Collaborative Decision Making (A-CDM) standards.

### 9.2 What Makes Production Systems Different

| Aspect | Course Project | Production System |
|--------|---------------|-------------------|
| **Cameras** | Single video file | 4-8 cameras per gate, 24/7 |
| **Model** | Pre-trained YOLO | Custom-trained on millions of gate images |
| **Accuracy** | Good enough to demonstrate | Must meet A-CDM standards (<60s error) |
| **Speed** | Offline processing | Real-time (30 FPS minimum) |
| **Infrastructure** | Your laptop | Edge GPU + cloud pipeline |
| **Output** | Console report | API integration with airline ops systems |

---

## 10. Connection to the Turnaround Project

Today's lecture gives you the intelligence layer for the turnaround calculator. Here's how everything from this week connects:

| Day | Technique | Role in Pipeline |
|-----|-----------|-----------------|
| **Day 1** | Image preprocessing | Clean each frame (resize, blur, color convert) |
| **Day 2** | Edge/contour analysis | Detect shapes and boundaries as fallback |
| **Day 3** | Background subtraction | Detect all motion, compute activity levels |
| **Day 3** | Object tracking | Maintain persistent IDs across frames |
| **Day 3** | Zone analysis | Focus on specific tarmac areas |
| **Day 4** | YOLO detection | Identify WHAT each object is (airplane, truck, person) |
| **Day 4** | YOLO + DeepSORT | Track identified objects with labels |
| **Day 4** | Event detection | Log turnaround events with timestamps |
| **Day 4** | Report generation | Calculate and display turnaround time |

### Your Project Approach Options

**Option A: Classical CV Only (Days 1-3)**
- Background subtraction + zone analysis + motion timeline
- Simpler, no deep learning needed
- Works if motion patterns are clear enough

**Option B: Deep Learning + Classical CV (Days 1-4)**
- YOLO detection + DeepSORT tracking + zone analysis
- More accurate, can identify specific equipment
- Requires GPU for reasonable speed

**Option C: Hybrid (Recommended)**
- Classical CV for motion detection and zone analysis
- YOLO for object identification on key frames
- Best of both worlds — accurate and efficient

---

## 11. Key Takeaways from Day 4

1. **Classical CV detects motion; deep learning identifies objects.** Combined, they give you both "something moved" and "it's a fuel truck."

2. **YOLO processes the entire image in one pass.** That's why it's fast enough for real-time video analysis — one forward pass gives all detections.

3. **Pre-trained YOLO knows 80 COCO classes.** For airport-specific equipment, custom training with labeled data is needed.

4. **Transfer learning is powerful.** Starting from a pre-trained model, you only need ~100 images per class to achieve good results on custom objects.

5. **The Ultralytics API is simple.** Three lines: load model, run inference, get results. Same API for detection, segmentation, and pose estimation.

6. **Combine YOLO with DeepSORT for persistent tracking.** YOLO detects per-frame, DeepSORT maintains identity across frames.

7. **Your turnaround pipeline combines everything from this week.** Preprocessing (Day 1) → Feature extraction (Day 2) → Motion & tracking (Day 3) → Intelligent detection (Day 4).

---

## Project Submission Reminder

Your final project is to build a CV system that calculates the turnaround time from `Airplane_Video.mp4`. Review the `PROJECT_README.md` for full requirements:

- **Minimum:** Detect arrival, detect departure, calculate turnaround time
- **Stretch:** Detect individual activities, annotated output video, visual timeline, web interface

Use any combination of techniques from this week. A simple approach that works is better than a complex system that doesn't. Document your thinking — the approach matters as much as the result.

```
your-submission/
├── README.md              # Your approach + results
├── requirements.txt       # Dependencies
├── src/
│   ├── main.py           # Entry point
│   ├── detector.py       # Detection logic
│   ├── tracker.py        # Tracking logic
│   └── utils.py          # Helpers
├── config/config.yaml    # Parameters
└── output/report.txt     # Generated report
```

---

**End of Day 4 Lecture Notes**
