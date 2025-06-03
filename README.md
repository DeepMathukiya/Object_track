# YOLOObjectTrack

**YOLOObjectTrack** is a C++ project for real-time **object detection**, **classification**, and **tracking** using:

- **YOLO** (You Only Look Once) for fast and accurate object detection.
- **SORT** (Simple Online and Realtime Tracking) with **Kalman Filter** and **kuhn munkres Algorithm** for robust multi-object tracking.
- **OpenCV** for video processing and visualization.
- **CUDA** acceleration for enhanced performance.

---

## 🚀 Features

- 🚦 Real-time object detection using YOLO.
- 🧭 Multi-object tracking using Kalman Filter + Hungarian Algorithm (SORT).
- 🧠 Modular design for easy customization and extension.
- 🖥️ GPU support via CUDA for high-performance inference.

---

## 🛠 Build Instructions

### 📋 Prerequisites

- CMake ≥ 3.5
- OpenCV (tested with 4.x)
- CUDA Toolkit ≥ 12
- C++17 compatible compiler

### 🧱 Build Steps

```bash
# Clone this repository
git clone https://github.com/DeepMathukiya/Object_track
cd Object_trac

# Create a build directory
mkdir build && cd build

# Configure the project
cmake ..

# Compile the executable
make

#Run the
./YOLOObjectTrack  (link of your desired camera) 

