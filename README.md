# 🖐️ Hand Gesture Master Volume Controller (MVC Architecture)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6.svg)
![Architecture](https://img.shields.io/badge/Architecture-MVC-purple.svg)

An intelligent, real-time computer vision application that controls the Windows system master audio volume based on the pinch distance between the user's **Thumb** and **Index finger**.

Built cleanly using the **Model-View-Controller (MVC)** architectural design pattern.

---

## 🌟 Key Features

- **Real-Time Hand Tracking**: Uses Google's MediaPipe Hands solution to detect and track key points at high FPS.
- **Euclidean Distance Math**: Computes the pixel distance between Thumb tip (Landmark `4`) and Index finger tip (Landmark `8`).
- **Smooth Audio Mapping**: Uses NumPy `interp` to map distance seamlessly into hardware decibels (`dB`) for Windows volume control.
- **Dynamic Graphical HUD**:
  - Highlights landmarks with colored indicators and connecting vectors.
  - Interactive visual indicator when fingers pinch closely.
  - On-screen vertical volume bar and percentage counter.
- **Clean MVC Architecture**: Fully decoupled logic separating computer vision models, hardware audio interface, UI drawing, and controllers.

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
    ├── models/              # Model Layer (Data & Hardware Logic)
    │   ├── __init__.py
    │   ├── audio_model.py        # Windows Audio hardware interface (Pycaw wrapper)
    │   └── hand_detector_model.py# MediaPipe hand tracking & landmark distance math
    ├── views/               # View Layer (GUI & Overlay Rendering)
    │   ├── __init__.py
    │   └── gui_view.py           # OpenCV frame capture, UI overlay & window display
    └── controllers/         # Controller Layer (Business Logic & Event Loop)
        ├── __init__.py
        └── main_controller.py    # Maps distances to volume & coordinates loop
```

---

## ⚡ Prerequisites & Installation

### Requirements
- **Operating System**: Windows (required for Pycaw COM interface)
- **Python**: Python 3.8+
- **Webcam**: Standard USB or Integrated Camera

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/FingerNumbers.git
   cd FingerNumbers
   ```

2. **Create and activate a virtual environment (Optional but Recommended)**:
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
- **Increase Volume**: Move thumb and index finger apart.
- **Decrease Volume**: Pinch thumb and index finger closer together.
- **Mute / Minimum Volume**: Touch thumb and index finger tips together (visual indicator turns green).
- **Quit Application**: Press the **`q`** key on your keyboard while focusing on the camera window.

---

## 📐 Math & Computer Vision Details

1. **Landmark Extraction**:
   - `Landmark 4`: Thumb Tip $(x_1, y_1)$
   - `Landmark 8`: Index Finger Tip $(x_2, y_2)$

2. **Euclidean Distance**:
   $$\text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

3. **Range Interpolation**:
   $$\text{Volume (dB)} = \text{np.interp}(\text{Distance}, [20, 200], [\text{min\_vol\_dB}, \text{max\_vol\_dB}])$$

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
