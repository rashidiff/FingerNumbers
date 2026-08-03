# 🖐️ Hand Finger Counter (MVC Architecture)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)
![Architecture](https://img.shields.io/badge/Architecture-MVC-purple.svg)

An intelligent, real-time computer vision application that detects and counts extended fingers (**0 to 5**) on a human hand using **OpenCV** and Google's **MediaPipe**.

Built cleanly following the **Model-View-Controller (MVC)** architectural design pattern.

---

## 🌟 Key Features

- **Real-Time Hand Tracking**: Uses Google's MediaPipe Hands solution for high FPS tracking on CPU.
- **Robust Detection**: Includes adaptive brightness/contrast enhancement to handle darker or backlit scenes without over-processing already well-lit frames.
- **Accurate Finger Counting (0 - 5)**:
  - Detects extended states for Thumb, Index, Middle, Ring, and Pinky fingers.
  - Uses 3D landmark geometry and joint-angle checks instead of only screen-space `y` comparisons.
- **Stable Output**: Applies a short temporal smoothing window to reduce flicker between adjacent counts.
- **Dynamic Graphical HUD**:
  - Highlights open finger tips with green indicators and closed tips with red indicators.
  - Large, clear visual counter box displaying the number of extended fingers.
- **Gesture Diagnostics**:
  - Shows a gesture label, bounding box, pose stability estimate, and session-level FPS/count metrics.
- **Clean MVC Architecture**: Fully modularized code separating computer vision algorithms, UI rendering, and event controllers.

---

## 🏗️ Architecture & Project Structure

The project strictly follows the **Model-View-Controller (MVC)** design pattern:

```text
FingerNumbers/
├── .gitignore               # Ignored files for Git version control
├── README.md                # Project documentation
├── requirements.txt         # Dependencies list
├── main.py                  # Application entry point
└── src/                     # Core application package
    ├── __init__.py
    ├── config.py             # Shared runtime configuration and thresholds
    ├── models/              # Model Layer (Data & Vision Logic)
    │   ├── __init__.py
    │   └── finger_counter_model.py # MediaPipe tracking & 0-5 finger counting math
    ├── views/               # View Layer (GUI & Overlay Rendering)
    │   ├── __init__.py
    │   └── gui_view.py           # OpenCV frame capture & Finger Count HUD rendering
    └── controllers/         # Controller Layer (Business Logic & Event Loop)
        ├── __init__.py
        └── main_controller.py    # Orchestrates vision model & view loop
└── tests/
    └── test_finger_counter_model.py # Unit tests for geometry and smoothing logic
```

---

## ⚡ Prerequisites & Installation

### Requirements
- **Python**: Python 3.8+
- **Webcam**: Standard USB or Integrated Camera

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/FingerNumbers.git
   cd FingerNumbers
   ```

2. **Create and activate a virtual environment (Optional)**:
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Run the main application script:

```bash
python main.py
```

### Controls & Gestures
- **Show Hand**: Hold your hand in front of the webcam.
- **Extend Fingers**: Raise 0, 1, 2, 3, 4, or 5 fingers to see the real-time count.
- **Quit Application**: Press the **`q`** key on your keyboard or click the **`X` (close)** button on the camera window to exit cleanly.
- **Keyboard Shortcuts**:
  - `d`: toggle diagnostics panel
  - `s`: toggle skeleton overlay
  - `h`: toggle controls hint
  - `r`: reset session stats and smoothing history

### Runtime Options

Launch with custom camera and tracking parameters:

```bash
python main.py --camera 1 --width 960 --height 540 --max-hands 2 --smoothing-window 7
```

Supported flags:
- `--camera`
- `--width`
- `--height`
- `--max-hands`
- `--detection-confidence`
- `--tracking-confidence`
- `--smoothing-window`
- `--hide-diagnostics`
- `--hide-controls`
- `--hide-skeleton`

### Run Tests

```bash
python -m unittest discover -s tests
```

---

## 📐 Finger Detection Logic

1. **Four Main Fingers (Index, Middle, Ring, Pinky)**:
   - Evaluates PIP joint angles in 3D landmark space.
   - Confirms that the fingertip extends farther from the wrist than the corresponding joint.

2. **Thumb Finger**:
   - Combines thumb joint angle with outward reach from the palm to avoid simple mirrored `x`-axis heuristics.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
