#!/usr/bin/env python3
"""
Face Recognition-Based Attendance Management System
---------------------------------------------------
Single integrated project: registration -> face capture -> LBPH training ->
real-time recognition -> attendance logging (SQLite) -> Excel export.

Detection  : Haar Cascade (OpenCV)
Recognition: LBPH (cv2.face.LBPHFaceRecognizer)
GUI        : Tkinter
"""
import contextlib
import csv
import datetime as dt
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import messagebox, ttk

# --------------------------------------------------------------------------
# Paths & configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
HAAR_DIR = os.path.join(BASE_DIR, "haarcascade")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
DB_DIR = os.path.join(BASE_DIR, "database")

CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml")
CASCADE_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/"
               "data/haarcascades/haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
DB_PATH = os.path.join(DB_DIR, "attendance.db")

EYE_CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_eye.xml")
LOG_PATH = os.path.join(ATTENDANCE_DIR, "event_log.csv")   # research event log

CAMERA_INDEX = 0
FACE_SIZE = (200, 200)        # every face crop is resized to this
DEFAULT_IMAGES = 40           # images captured per student (30-50)
UNKNOWN_THRESHOLD = 70.0      # LBPH distance above this => "Unknown"
CONSECUTIVE_FRAMES = 3        # frames a face must be recognised in a row
COOLDOWN_SECONDS = 5          # min seconds between attempts for one student
BLINK_MAX_CLOSED_FRAMES = 6   # eyes "closed" for 1..6 frames then reopen = a blink
BLINK_WINDOW_SECONDS = 5      # a blink counts for this long
SCENARIOS = ("normal", "genuine_trial", "printed_photo", "screen_replay", "stranger")


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def ensure_dirs():
    for d in (DATASET_DIR, TRAINER_DIR, HAAR_DIR, ATTENDANCE_DIR, DB_DIR):
        os.makedirs(d, exist_ok=True)


def ensure_cascade():
    """Make sure the Haar Cascade XMLs exist (copy from OpenCV or download)."""
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_eye.xml"):
        target = os.path.join(HAAR_DIR, name)
        if os.path.exists(target):
            continue
        bundled = os.path.join(cv2.data.haarcascades, name)
        if os.path.exists(bundled):
            shutil.copy(bundled, target)
        else:
            urllib.request.urlretrieve(CASCADE_URL.replace(
                "haarcascade_frontalface_default.xml", name), target)


_cascade = None


def detect_faces(gray):
    """Detect faces in a grayscale frame with the Haar Cascade."""
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if _cascade.empty():
            raise RuntimeError("Could not load Haar Cascade: " + CASCADE_PATH)
    return _cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                     minSize=(80, 80))


def preprocess(gray_face):
    """Same preprocessing for training and recognition (resize + equalize)."""
    return cv2.equalizeHist(cv2.resize(gray_face, FACE_SIZE))


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
@contextlib.contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY AUTOINCREMENT,   -- also the LBPH label
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT, year TEXT, section TEXT, roll_no TEXT,
            registered_on TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_pk INTEGER NOT NULL,
            student_id TEXT, name TEXT,
            date TEXT NOT NULL, time TEXT NOT NULL,
            UNIQUE(student_pk, date))""")  # one record per student per day


def add_student(student_id, name, department, year, section, roll_no):
    with db() as con:
        cur = con.execute(
            "INSERT INTO students(student_id,name,department,year,section,roll_no,"
            "registered_on) VALUES(?,?,?,?,?,?,?)",
            (student_id, name, department, year, section, roll_no,
             dt.datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid


def list_students():
    with db() as con:
        return con.execute("SELECT id,student_id,name,department,year,section,roll_no "
                           "FROM students ORDER BY student_id").fetchall()


def get_student(pk):
    with db() as con:
        return con.execute("SELECT id,student_id,name FROM students WHERE id=?",
                           (pk,)).fetchone()


def student_dir(pk, name):
    """dataset/<pk>_<name>/ -- the numeric prefix is the LBPH label."""
    return os.path.join(DATASET_DIR, f"{pk}_{re.sub(r'[^A-Za-z0-9]+', '_', name)}")


def count_images(pk, name):
    d = student_dir(pk, name)
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def delete_student(pk):
    row = get_student(pk)
    if not row:
        return
    with db() as con:
        con.execute("DELETE FROM students WHERE id=?", (pk,))
    shutil.rmtree(student_dir(pk, row[2]), ignore_errors=True)


def mark_attendance(pk):
    """Insert today's record. Returns ('marked'|'duplicate'|'missing', name)."""
    now = dt.datetime.now()
    row = get_student(pk)
    if not row:
        return "missing", None
    with db() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO attendance(student_pk,student_id,name,date,time) "
            "VALUES(?,?,?,?,?)",
            (pk, row[1], row[2], now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
        return ("marked" if cur.rowcount == 1 else "duplicate"), row[2]


def log_event(session, scenario, liveness, label, name, dist, status):
    """Append one row to attendance/event_log.csv (used by analyze_trials.py)."""
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["session", "timestamp", "scenario", "liveness",
                        "student", "name", "distance", "status"])
        w.writerow([session, dt.datetime.now().isoformat(timespec="seconds"), scenario,
                    int(liveness), label, name, "" if dist is None else round(dist, 1), status])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_model():
    """Train LBPH on every image in dataset/. Returns (n_images, n_students)."""
    faces, labels = [], []
    for folder in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(path):
            continue
        try:
            label = int(folder.split("_")[0])
        except ValueError:
            continue
        for fname in os.listdir(path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = np.array(Image.open(os.path.join(path, fname)).convert("L"),
                               dtype="uint8")
                faces.append(preprocess(img))
                labels.append(label)
    if not faces:
        raise ValueError("No face images found. Capture faces first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8,
                                                    grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.write(MODEL_PATH)
    return len(faces), len(set(labels))


# --------------------------------------------------------------------------
# Recognition engine (GUI independent)
# --------------------------------------------------------------------------
class AttendanceEngine:
    def __init__(self, liveness=False, scenario="normal"):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not trained yet. Click 'Train Model'.")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(MODEL_PATH)
        self.names = {r[0]: r[2] for r in list_students()}
        self.streak = {}      # label -> consecutive recognised frames
        self.last_try = {}    # label -> time of last attendance attempt
        self.liveness = liveness            # require a blink before marking
        self.scenario = scenario            # "normal" writes to DB; others only log
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH) if liveness else None
        self.eye_state = {}   # label -> blink tracking
        self.last_pending_log = {}
        self.session = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_event(self.session, scenario, liveness, "", "", None, "session_start")

    def update_blink(self, label, face_gray, now):
        """Heuristic blink detector: eyes visible -> briefly not visible -> visible."""
        st = self.eye_state.setdefault(label, {"open": 0, "closed": 0, "blink_t": 0.0})
        upper = face_gray[:int(face_gray.shape[0] * 0.6)]      # eyes are in the top part
        eyes = self.eye_cascade.detectMultiScale(upper, 1.1, 5, minSize=(15, 15))
        if len(eyes) >= 1:
            if 1 <= st["closed"] <= BLINK_MAX_CLOSED_FRAMES and st["open"] >= 2:
                st["blink_t"] = now                              # blink completed
            st["open"] += 1
            st["closed"] = 0
        else:
            st["closed"] += 1
            if st["closed"] > BLINK_MAX_CLOSED_FRAMES:           # too long: not a blink
                st["open"] = 0

    def blink_recent(self, label, now):
        return now - self.eye_state.get(label, {}).get("blink_t", 0.0) <= BLINK_WINDOW_SECONDS

    def process(self, frame):
        """Annotate frame; return (frame, list of (status, name))."""
        events, seen, dists = [], set(), {}
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detect_faces(gray):
            roi = gray[y:y + h, x:x + w]
            label, dist = self.recognizer.predict(preprocess(roi))
            if dist <= UNKNOWN_THRESHOLD and label in self.names:
                text, color = f"{self.names[label]} ({dist:.0f})", (0, 200, 0)
                seen.add(label)
                dists[label] = dist
                self.streak[label] = self.streak.get(label, 0) + 1
                if self.liveness:
                    self.update_blink(label, roi, now)
                    if not self.blink_recent(label, now):
                        text += " - please blink"
            else:
                text, color = "Unknown", (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for label in list(self.streak):     # streak must be *consecutive*
            if label not in seen:
                self.streak[label] = 0

        for label in seen:
            if (self.streak[label] >= CONSECUTIVE_FRAMES
                    and now - self.last_try.get(label, 0) >= COOLDOWN_SECONDS):
                name = self.names[label]
                if self.liveness and not self.blink_recent(label, now):
                    # recognised but no blink yet: wait (streak kept, no cooldown started)
                    if now - self.last_pending_log.get(label, 0) >= COOLDOWN_SECONDS:
                        self.last_pending_log[label] = now
                        log_event(self.session, self.scenario, True, label, name,
                                  dists[label], "liveness_pending")
                    continue
                self.last_try[label] = now          # start cooldown
                self.streak[label] = 0
                if self.scenario == "normal":
                    status, name = mark_attendance(label)
                else:                               # research trial: do not touch the DB
                    status = "would_mark"
                log_event(self.session, self.scenario, self.liveness, label, name,
                          dists[label], status)
                events.append((status, name))
        return frame, events


# --------------------------------------------------------------------------
# Excel export
# --------------------------------------------------------------------------
def export_excel(start, end):
    """Export attendance between two dates (YYYY-MM-DD). Returns path or None."""
    with db() as con:
        df = pd.read_sql_query(
            'SELECT a.date AS "Date", a.time AS "Time", a.student_id AS "Student ID", '
            'a.name AS "Name", s.department AS "Department", s.year AS "Year", '
            's.section AS "Section", s.roll_no AS "Roll No" '
            "FROM attendance a LEFT JOIN students s ON s.id=a.student_pk "
            "WHERE a.date BETWEEN ? AND ? ORDER BY a.date, a.time",
            con, params=(start, end))
    if df.empty:
        return None
    summary = (df.groupby(["Student ID", "Name"]).size()
                 .reset_index(name="Days Present"))
    path = os.path.join(ATTENDANCE_DIR, "attendance_%s_to_%s_%s.xlsx" % (
        start, end, dt.datetime.now().strftime("%H%M%S")))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Attendance", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for sheet, data in (("Attendance", df), ("Summary", summary)):
            ws = writer.sheets[sheet]       # auto-fit column widths
            for i, col in enumerate(data.columns, start=1):
                width = max(len(str(col)), *(len(str(v)) for v in data[col])) + 2
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    return path


# --------------------------------------------------------------------------
# GUI: camera windows
# --------------------------------------------------------------------------
class CameraWindow(tk.Toplevel):
    """Base window: pulls webcam frames and shows them in a Label."""

    def __init__(self, app, title, cap):
        super().__init__(app)
        self.app, self.cap, self.running = app, cap, True
        self.title(title)
        self.video = tk.Label(self)
        self.video.pack()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def tick(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = self.handle(frame)
            if not self.running:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=self.photo)
        self.after(15, self.tick)

    def close(self):
        if not self.running:
            return
        self.running = False
        self.cap.release()
        self.destroy()
        self.app.camera_closed()


class CaptureWindow(CameraWindow):
    """Collects face crops for one student and saves them to dataset/."""

    def __init__(self, app, cap, student, target):
        super().__init__(app, "Capture Faces - " + student[2], cap)
        self.student, self.target, self.count, self.skip = student, target, 0, 0
        self.capturing = False
        self.folder = student_dir(student[0], student[2])
        ttk.Button(self, text="Start Capture", command=self.start).pack(pady=4)
        self.tick()

    def start(self):
        shutil.rmtree(self.folder, ignore_errors=True)   # replace old images
        os.makedirs(self.folder)
        self.count, self.capturing = 0, True

    def handle(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 160, 0), 2)
            if self.capturing:
                self.skip += 1
                if self.skip % 3 == 0:           # small gap => varied poses
                    self.count += 1
                    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                    cv2.imwrite(os.path.join(self.folder, f"img_{self.count:03d}.jpg"), face)
        msg = (f"Captured {self.count}/{self.target}" if self.capturing
               else "Press 'Start Capture' - move head slightly")
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if self.capturing and self.count >= self.target:
            self.app.set_status(f"Saved {self.count} images for {self.student[2]}")
            messagebox.showinfo("Done", f"{self.count} images saved.\n"
                                        "Now click 'Train Model'.", parent=self)
            self.close()
        return frame


class AttendanceWindow(CameraWindow):
    """Live recognition + attendance logging."""

    def __init__(self, app, cap, engine):
        super().__init__(app, "Live Attendance", cap)
        self.engine = engine
        self.log = tk.Listbox(self, height=6, width=70)
        self.log.pack(padx=4, pady=4)
        ttk.Button(self, text="Stop Attendance", command=self.close).pack(pady=4)
        self.tick()

    def handle(self, frame):
        frame, events = self.engine.process(frame)
        for status, name in events:
            stamp = dt.datetime.now().strftime("%H:%M:%S")
            text = {"marked": f"{stamp}  {name} - attendance MARKED",
                    "duplicate": f"{stamp}  {name} - already marked today",
                    "would_mark": f"{stamp}  {name} - WOULD mark (trial mode)",
                    "missing": f"{stamp}  unknown record"}[status]
            self.log.insert(0, text)
        return frame


# --------------------------------------------------------------------------
# GUI: main admin window
# --------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("960x640")
        self.cam_window = None
        self.train_queue = queue.Queue()
        self.vars = {k: tk.StringVar() for k in
                     ("student_id", "name", "department", "year", "section", "roll_no")}
        self.n_images = tk.IntVar(value=DEFAULT_IMAGES)
        today = dt.date.today().isoformat()
        self.from_var, self.to_var = tk.StringVar(value=today), tk.StringVar(value=today)
        self.liveness = tk.BooleanVar(value=False)
        self.scenario = tk.StringVar(value="normal")
        self.status = tk.StringVar(value="Ready")
        self.build_ui()
        self.refresh_table()

    # ---- layout ----
    def build_ui(self):
        left = ttk.LabelFrame(self, text="1. Register Student")
        left.pack(side="left", fill="y", padx=8, pady=8)
        labels = [("Student ID", "student_id"), ("Name", "name"),
                  ("Department", "department"), ("Year", "year"),
                  ("Section", "section"), ("Roll Number", "roll_no")]
        for i, (text, key) in enumerate(labels):
            ttk.Label(left, text=text).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(left, textvariable=self.vars[key], width=24).grid(row=i, column=1, padx=6)
        ttk.Button(left, text="Register Student", command=self.register).grid(
            row=6, column=0, columnspan=2, pady=8)

        ttk.Separator(left).grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(left, text="Images per student").grid(row=8, column=0, sticky="w", padx=6)
        ttk.Spinbox(left, from_=30, to=50, textvariable=self.n_images, width=6).grid(
            row=8, column=1, sticky="w", padx=6)

        ttk.Separator(left).grid(row=9, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(left, text="Export Excel (YYYY-MM-DD)").grid(row=10, column=0, columnspan=2)
        ttk.Label(left, text="From").grid(row=11, column=0, sticky="w", padx=6)
        ttk.Entry(left, textvariable=self.from_var, width=16).grid(row=11, column=1, pady=2)
        ttk.Label(left, text="To").grid(row=12, column=0, sticky="w", padx=6)
        ttk.Entry(left, textvariable=self.to_var, width=16).grid(row=12, column=1, pady=2)
        ttk.Button(left, text="Export Excel Report", command=self.export).grid(
            row=13, column=0, columnspan=2, pady=8)

        ttk.Separator(left).grid(row=14, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Checkbutton(left, text="Require blink (liveness)", variable=self.liveness).grid(
            row=15, column=0, columnspan=2, sticky="w", padx=6)
        ttk.Label(left, text="Trial scenario").grid(row=16, column=0, sticky="w", padx=6)
        ttk.Combobox(left, textvariable=self.scenario, values=SCENARIOS, width=14,
                     state="readonly").grid(row=16, column=1, padx=6, pady=2)

        right = ttk.LabelFrame(self, text="Registered Students (select one to capture faces)")
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        cols = ("ID", "Name", "Dept", "Year", "Section", "Roll", "Images")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=14)
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=90 if c != "Name" else 150)
        self.table.pack(fill="both", expand=True, padx=6, pady=6)

        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=4)
        for text, cmd in (("2. Capture Faces", self.capture),
                          ("3. Train Model", self.train),
                          ("4. Start Attendance", self.start_attendance),
                          ("Stop Attendance", self.stop_attendance),
                          ("Delete Student", self.remove)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=3)
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

    # ---- helpers ----
    def set_status(self, text):
        self.status.set(text)

    def refresh_table(self):
        self.table.delete(*self.table.get_children())
        for pk, sid, name, dept, year, sec, roll in list_students():
            self.table.insert("", "end", iid=str(pk),
                              values=(sid, name, dept, year, sec, roll, count_images(pk, name)))

    def selected_student(self):
        sel = self.table.selection()
        if not sel:
            messagebox.showwarning("Select student", "Select a student in the table first.")
            return None
        return get_student(int(sel[0]))

    def open_camera(self):
        if self.cam_window is not None:
            messagebox.showinfo("Camera busy", "Close the current camera window first.")
            return None
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open the webcam.")
            return None
        return cap

    def camera_closed(self):
        self.cam_window = None
        self.refresh_table()
        self.set_status("Ready")

    # ---- actions ----
    def register(self):
        v = {k: var.get().strip() for k, var in self.vars.items()}
        if not v["student_id"] or not v["name"]:
            messagebox.showwarning("Missing data", "Student ID and Name are required.")
            return
        try:
            pk = add_student(**v)
        except sqlite3.IntegrityError:
            messagebox.showerror("Duplicate", "That Student ID is already registered.")
            return
        for var in self.vars.values():
            var.set("")
        self.refresh_table()
        self.table.selection_set(str(pk))
        self.set_status(f"Registered {v['name']}. Click 'Capture Faces' next.")

    def capture(self):
        student = self.selected_student()
        if not student:
            return
        target = max(30, min(50, int(self.n_images.get())))
        cap = self.open_camera()
        if cap:
            self.cam_window = CaptureWindow(self, cap, student, target)

    def train(self):
        self.set_status("Training model... please wait")

        def work():
            try:
                self.train_queue.put(("ok", train_model()))
            except Exception as exc:          # report any error to the GUI
                self.train_queue.put(("err", str(exc)))
        threading.Thread(target=work, daemon=True).start()
        self.after(200, self.poll_training)

    def poll_training(self):
        try:
            kind, data = self.train_queue.get_nowait()
        except queue.Empty:
            self.after(200, self.poll_training)
            return
        if kind == "ok":
            self.set_status(f"Model trained: {data[0]} images, {data[1]} student(s)")
            messagebox.showinfo("Training complete",
                                f"Trained on {data[0]} images of {data[1]} student(s).")
        else:
            self.set_status("Training failed")
            messagebox.showerror("Training failed", data)

    def start_attendance(self):
        try:
            engine = AttendanceEngine(self.liveness.get(), self.scenario.get())
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        cap = self.open_camera()
        if cap:
            self.cam_window = AttendanceWindow(self, cap, engine)
            self.set_status("Attendance running (scenario=%s, liveness=%s)" % (
                self.scenario.get(), self.liveness.get()))

    def stop_attendance(self):
        if isinstance(self.cam_window, AttendanceWindow):
            self.cam_window.close()
        else:
            self.set_status("Attendance is not running")

    def remove(self):
        student = self.selected_student()
        if student and messagebox.askyesno(
                "Delete", f"Delete {student[2]} and their face images?\n"
                          "(Past attendance records are kept.) Retrain afterwards."):
            delete_student(student[0])
            self.refresh_table()

    def export(self):
        start, end = self.from_var.get().strip(), self.to_var.get().strip()
        try:
            dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        except ValueError:
            messagebox.showerror("Date", "Use the format YYYY-MM-DD.")
            return
        path = export_excel(start, end)
        if path:
            self.set_status("Exported: " + path)
            messagebox.showinfo("Export complete", "Saved to:\n" + path)
        else:
            messagebox.showinfo("No data", "No attendance records in that range.")


if __name__ == "__main__":
    ensure_dirs()
    ensure_cascade()
    init_db()
    App().mainloop()