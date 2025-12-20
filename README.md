This repository contains the backend and verification logic for the Green-ID sustainability platform. The project is divided into four main technical modules, each handling a specific type of eco-friendly activity verification using computer vision, geolocation algorithms, and deep learning.

## 🛠 Features & Technology Stack

### 1. Cleanup Verification (`cleanup/`)
**Technology:** Python, Flask, Ultralytics YOLOv8 (Classification)

The cleanup module creates an API endpoint to verify waste removal tasks.
- **Algorithm:** Utilizes a **fine-tuned YOLOv8 classification model** (`best.pt`) specifically trained to distinguish between high-entropy "before" states (trash present) and organized "after" states (clean).
- **Endpoint:** `/verify_cleanup` accepts paired "before" and "after" images.
- **Logic:** The system runs inference on both images independently and asserts that the first image is classified as 'before' and the second as 'after' with high confidence.

### 2. Walking Verification (`walking/`)
**Technology:** Python, Flask, Haversine Formula

A geospatial verification service for tracking physical activity.
- **Algorithm:** Implements the **Haversine formula** to calculate the precise great-circle distance between a series of GPS coordinates (latitude/longitude pairs).
- **Logic:** The system iterates through a stream of timestamped coordinate points, summing the geodesic path segments.
- **Validation:** A walk is marked as valid only if the calculated total distance exceeds a configurable threshold (default: 2.0 km).

### 3. Planting Verification (`planting/`)
**Technology:** Python, OpenCV, Ultralytics YOLOv8 (Object Detection)

A real-time computer vision verification system for tree/sapling planting activities.
- **Algorithm:** Hybrid detection pipeline.
    - **Primary:** Uses **YOLOv8n** (Nano) for object detection to simultaneously identify classes: `person` and `plant` (including trees/potted plants).
    - **Fallback:** Implements a legacy Computer Vision fallback using **HSV color space thresholding** to detect significant green biomass ("Green Blob") if the neural network confidence is low.
- **Usage:** Capable of running on live webcam feeds or pre-recorded footage.
- **Note:** *The planting module includes a demo video mode for consistent testing and demonstration purposes.*

### 4. Public Transport Recognition (`PublicTransport/`)
**Technology:** Python, TensorFlow 2.x, Keras, NumPy

Deep learning classifier to validate usage of public transit systems.
- **Architecture:** MobileNetV2 (CNN) built with Keras.
- **Preprocessing:** Input images are resized to `(128, 128)` and normalized.
- **Logic:** The model analyzes user-uploaded photos to classify them into specific public transport categories (e.g., Bus, Metro, Auto-rickshaw).
- **Inference:** Outputs a predicted class label and a confidence score percentage.

---
**Dependencies:**
- `ultralytics` (YOLO)
- `tensorflow`
- `flask`
- `opencv-python`
- `numpy`
