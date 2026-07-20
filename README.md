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
- **Accurate Finger Counting (0 - 5)**:
  - Detects extended states for Thumb, Index, Middle, Ring, and Pinky fingers.
  - Handles left and right hand orientation checks dynamically.
- **Dynamic Graphical HUD**:
  - Highlights open finger tips with green indicators and closed tips with red indicators.
  - Large, clear visual counter box displaying the number of extended fingers.
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
    ├── models/              # Model Layer (Data & Vision Logic)
    │   ├── __init__.py
    │   └── finger_counter_model.py # MediaPipe tracking & 0-5 finger counting math
    ├── views/               # View Layer (GUI & Overlay Rendering)
    │   ├── __init__.py
    │   └── gui_view.py           # OpenCV frame capture & Finger Count HUD rendering
    └── controllers/         # Controller Layer (Business Logic & Event Loop)
        ├── __init__.py
        └── main_controller.py    # Orchestrates vision model & view loop
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
- **Quit Application**: Press the **`q`** key on your keyboard while focusing on the camera window.

---

## 📐 Finger Detection Logic

1. **Four Main Fingers (Index, Middle, Ring, Pinky)**:
   - Evaluates vertical landmark coordinates ($y_{tip} < y_{pip}$).
   - If finger tip is higher in the frame than PIP joint, finger is counted as extended (`1`).

2. **Thumb Finger**:
   - Evaluates horizontal landmark coordinates ($x_{tip}$ vs $x_{pip}$) adapted to hand orientation (`Right` or `Left`).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
