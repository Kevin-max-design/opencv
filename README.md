# 🎨 Air Drawing — Touchless Gesture-Based Drawing System

A real-time touchless drawing application using **hand gestures** and **computer vision**, powered entirely by your webcam. Draw in the air by pinching your fingers — no mouse, keyboard, or touchscreen required.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| ✍️ **Pinch-to-Draw** | Bring index finger and thumb together to start drawing |
| 🎨 **Color Palette** | 8 vibrant colors selectable via gesture |
| 📏 **Brush Sizes** | 4 thickness options (S, M, L, XL) |
| 🧹 **Clear Canvas** | Virtual CLEAR button — touch-free |
| ↩️ **Undo Support** | Undo last stroke via gesture or keyboard |
| 🔄 **Temporal Smoothing** | EMA-based jitter elimination for human-like strokes |
| ⚡ **Gesture Debouncing** | Prevents false triggers from noisy hand motion |
| 📊 **Debug Mode** | Real-time pinch distance, hand size, and state visualization |
| 🖥️ **FPS Counter** | Live performance monitoring |

---

## 🧠 How It Works

```
Webcam Frame → Hand Tracking (MediaPipe) → Gesture Recognition (Pinch Detection)
                                                      ↓
                                              State Machine (Debounced)
                                              ↓              ↓
                                          Drawing         UI Interaction
                                        (Smoothed)      (Color/Size/Clear)
                                              ↓              ↓
                                         Canvas Overlay ← Updated State
                                              ↓
                                    Composited Output Frame
```

### Key Technical Challenges Solved

1. **Noisy Hand Motion** → Exponential Moving Average (EMA) + weighted position history
2. **Jittery Pinch Detection** → Dynamic threshold scaled by hand size
3. **False Triggers** → N-frame debouncing with hysteresis state machine
4. **Real-time Performance** → Optimized pipeline with minimal per-frame allocations

---

## ⚙️ Tech Stack

- **Python 3.8+**
- **OpenCV** — Webcam capture, drawing primitives, display
- **MediaPipe** — Real-time hand landmark detection (21 landmarks)
- **NumPy** — Canvas operations, vector math, smoothing

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam

### Installation

```bash
# Clone or navigate to the project directory
cd air-drawing

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
python main.py
```

---

## 🎮 Controls

### Gesture Controls
| Gesture | Action |
|---------|--------|
| 🤏 Pinch (index + thumb) | Start drawing |
| ✋ Release pinch | Stop drawing |
| 🤏 Pinch over color swatch | Change drawing color |
| 🤏 Pinch over size button | Change brush size |
| 🤏 Pinch over CLEAR | Clear the canvas |
| 🤏 Pinch over UNDO | Undo last stroke |

### Keyboard Controls
| Key | Action |
|-----|--------|
| `q` | Quit application |
| `c` | Clear canvas |
| `u` | Undo last stroke |
| `d` | Toggle debug overlay |
| `l` | Toggle hand landmark display |

---

## 📁 Project Structure

```
air-drawing/
├── main.py               # Application entry point & main loop
├── hand_tracker.py        # MediaPipe hand tracking wrapper
├── gesture_recognizer.py  # Pinch detection & state machine
├── drawing_engine.py      # Canvas, smoothing, stroke management
├── ui_overlay.py          # Virtual toolbar rendering
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🏗️ Architecture

| Module | Responsibility |
|--------|---------------|
| `HandTracker` | Wraps MediaPipe Hands; extracts key landmarks in pixel coords |
| `GestureRecognizer` | Debounced pinch detection with IDLE/DRAWING/UI state machine |
| `DrawingEngine` | Manages BGRA canvas; temporal smoothing; stroke history & undo |
| `UIOverlay` | Renders toolbar; hit-tests button regions for gesture interaction |
| `AirDrawingApp` | Orchestrates pipeline; composites all layers; handles I/O |

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
