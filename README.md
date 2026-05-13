# 🎯 Real-Time Face Detection & Recognition System

A real-time face detection and recognition system built with **C++** and **OpenCV 4.12**, 
capable of identifying known individuals and flagging unknown faces using a live webcam feed.

---

## 📸 Demo

> Green box = Known person | Red box = Unknown person

<!-- Add your screenshot here after uploading -->
![Demo Screenshot](screenshot_1778704065.jpg)

---

## ✨ Features

- 🟢 Real-time face detection via webcam
- 🟢 Known person → **Green box + Name label**
- 🔴 Unknown person → **Red box + "Unknown" label**
- 📊 Live FPS counter
- 📷 Press `S` to save screenshot
- 🧠 LBPH Face Recognition algorithm
- ➕ Easily add new people by dropping photos into a folder

---

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| Language | C++ 17 |
| Computer Vision | OpenCV 4.12 |
| Face Recognition | opencv_contrib (LBPH) |
| Compiler | MSVC v142 (VS2019) |
| Package Manager | vcpkg |
| IDE | VS Code |
| OS | Windows 10/11 |

---

## 📁 Project Structure

face-recognition-opencv/
├── src/
│   └── main.cpp              # Main application source
├── data/
│   ├── known_faces/          # Add person folders here
│   │   └── YourName/
│   │       ├── 1.jpg
│   │       └── 2.jpg
│   └── haarcascade_frontalface_default.xml
├── .vscode/
│   ├── c_cpp_properties.json
│   ├── tasks.json
│   └── launch.json
└── README.md

---

## ⚙️ Prerequisites

- Windows 10/11
- Visual Studio 2019 (with C++ workload)
- CMake 4.x
- Git
- vcpkg

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-opencv.git
cd face-recognition-opencv
```

### 2. Install OpenCV via vcpkg
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install opencv4[contrib,dnn,world]:x64-windows
```

### 3. Add your face photos

data/known_faces/YourName/1.jpg
data/known_faces/YourName/2.jpg

### 4. Build
```cmd
cl.exe /EHsc /std:c++17 /MD ^
  /I "D:/vcpkg/installed/x64-windows/include" ^
  /I "D:/vcpkg/installed/x64-windows/include/opencv4" ^
  src/main.cpp ^
  /link ^
  /LIBPATH:"D:/vcpkg/installed/x64-windows/lib" ^
  opencv_world4.lib ^
  /OUT:FaceRecognition.exe
```

### 5. Run
```cmd
set PATH=D:\vcpkg\installed\x64-windows\bin;%PATH%
FaceRecognition.exe
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit the application |
| `S` | Save screenshot |

---

## 🧠 How It Works

1. **Detection** — Haar Cascade detects face regions in each frame
2. **Recognition** — LBPH algorithm compares detected face against trained data
3. **Decision** — Confidence score below threshold → Known person (green)
4. **Display** — Name label drawn above bounding box in real time

---

## 📌 Confidence Threshold

The default threshold is `80.0`. You can tune it in `main.cpp`:
```cpp
const double CONFIDENCE_THRESHOLD = 80.0;
// Lower = stricter | Higher = more lenient
```

---

## 👨‍💻 Author

**Yukesh** — Built as a computer vision learning project.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)