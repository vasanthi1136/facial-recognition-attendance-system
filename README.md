# Face Recognition-Based Attendance Management System

Real-time face recognition attendance system using OpenCV (Haar Cascade detection + LBPH recognition) with a Tkinter GUI. Marks attendance once per student per day in SQLite after 3 consecutive recognised frames, and exports date-range Excel reports.

## Features
- Register students and capture face images from the webcam
- Train an LBPH face recognition model
- Real-time recognition: green box for known students, "Unknown" for others
- Attendance marked after 3 consecutive recognised frames, once per student per day
- Export attendance reports to Excel (Attendance and Summary sheets)

## Tech Stack
Python, OpenCV, SQLite, Tkinter, Pandas, NumPy, Pillow, openpyxl

## Installation
1. Install Python 3.8+ (on Linux also: `sudo apt install python3-tk`)
2. Install libraries: `pip install -r requirements.txt`
3. Run: `python main.py`

## How to Use
1. Register a student in the left panel
2. Select the student and click "Capture Faces"
3. Click "Train Model"
4. Click "Start Attendance"
5. Enter From/To dates and click "Export Excel Report"

## Troubleshooting
If you see `module 'cv2' has no attribute 'face'`:

```
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```
