FACE RECOGNITION-BASED ATTENDANCE MANAGEMENT SYSTEM
====================================================
Haar Cascade (detection) + LBPH (recognition) + SQLite + Excel reports + Tkinter GUI.

FOLDER STRUCTURE
  face_attendance/
    dataset/      face images, one sub-folder per student (<id>_<name>/)
    trainer/      trained model (trainer.yml)
    haarcascade/  haarcascade_frontalface_default.xml
    attendance/   exported Excel reports
    database/     attendance.db (SQLite)
    main.py       the whole application
    requirements.txt

INSTALLATION
  1. Install Python 3.8+ (on Linux also: sudo apt install python3-tk).
  2. Open a terminal inside the face_attendance folder and run:
       pip install -r requirements.txt
     (or: pip install opencv-contrib-python numpy Pillow pandas openpyxl)
  3. The Haar Cascade XML is bundled. If missing, main.py restores it
     automatically, or download it manually:
       https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
     and save it into the haarcascade/ folder.
  4. Run:  python main.py      (Linux/macOS: python3 main.py)

HOW TO REGISTER A STUDENT
  Fill in Student ID, Name, Department, Year, Section, Roll Number in the left
  panel and click "Register Student". (ID and Name are required; ID must be unique.)
  The student appears in the table on the right.

HOW TO CAPTURE FACE IMAGES
  1. Select the student in the table.
  2. (Optional) set "Images per student" (30-50).
  3. Click "2. Capture Faces", then "Start Capture" in the camera window.
  4. Face the camera with good lighting and slowly turn your head a little.
     The window closes automatically when done. Re-capturing replaces old images.

HOW TO TRAIN THE MODEL
  Click "3. Train Model" after capturing faces for all students. Re-train whenever
  students are added, deleted or re-captured. For reliable "Unknown" detection,
  register at least two students.

HOW TO START ATTENDANCE
  Click "4. Start Attendance". The webcam opens; recognised students get a green box,
  others show "Unknown" (LBPH distance > 70). Attendance is marked only when a student
  is recognised in 3 consecutive frames, with a 5-second cooldown, and only once per
  student per day. Click "Stop Attendance" (or close the window) to stop.

HOW TO EXPORT EXCEL REPORTS
  Enter From/To dates (YYYY-MM-DD; default is today) and click "Export Excel Report".
  The .xlsx (sheets: Attendance, Summary) is saved in the attendance/ folder.

TUNING (constants at the top of main.py)
  UNKNOWN_THRESHOLD (70), CONSECUTIVE_FRAMES (3), COOLDOWN_SECONDS (5),
  CAMERA_INDEX (0; try 1 for an external webcam).

TROUBLESHOOTING
  - "module 'cv2' has no attribute 'face'": uninstall other OpenCV builds, then
      pip uninstall opencv-python opencv-contrib-python
      pip install opencv-contrib-python
  - Wrong people recognised: capture in similar lighting to the classroom, retrain,
    or lower UNKNOWN_THRESHOLD. Many false "Unknown": raise it slightly (e.g. 80).

Then ---> Go for training the model on kaggle dataset to train the  model
 
pip install scikit-learn matplotlib

python prepare_benchmark.py --name olivetti --out C:\data\orl


python research.py --data-dir C:\data\orl


python prepare_benchmark.py --name lfw --out C:\data\lfw --min-faces 20
python research.py --data-dir C:\data\lfw --detect
python evaluate.py --data-dir C:\data\lfw --detect --split random

python baselines.py --data-dir C:\data\orl --repeats 10
python baselines.py --data-dir C:\data\lfw --repeats 5 --max-images 50


