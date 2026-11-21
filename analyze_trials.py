#!/usr/bin/env python3
"""
Summarise live trials from attendance/event_log.csv.

How to collect data (one 'session' = one Start Attendance ... Stop Attendance run):
  1. In the GUI choose a Trial scenario (genuine_trial / printed_photo / screen_replay /
     stranger) and tick/untick 'Require blink'.   (scenario != normal never writes to the DB)
  2. Start Attendance, present the face / photo / phone for ~10-15 s, Stop Attendance.
  3. Repeat for many sessions per condition (aim for 20+), with and without liveness.

Then run:  python analyze_trials.py
Success rate = share of sessions in which the system would have marked attendance.
"""
import math
import os

import pandas as pd

import main as M


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def main():
    if not os.path.exists(M.LOG_PATH):
        raise SystemExit("No event log yet: " + M.LOG_PATH)
    df = pd.read_csv(M.LOG_PATH)
    sessions = df[df.status == "session_start"][["session", "scenario", "liveness"]]
    marked = df[df.status.isin(["marked", "would_mark", "duplicate"])].session.unique()
    pending = df[df.status == "liveness_pending"].session.unique()
    sessions = sessions.assign(
        success=sessions.session.isin(marked),
        blocked=sessions.session.isin(pending) & ~sessions.session.isin(marked))
    rows = []
    for (sc, lv), g in sessions.groupby(["scenario", "liveness"]):
        n, k = len(g), int(g.success.sum())
        lo, hi = wilson(k, n)
        rows.append({"Scenario": sc, "Liveness": "on" if lv else "off", "Sessions": n,
                     "Marked": k, "Success %": round(k / n * 100, 1),
                     "95% CI": "%.1f-%.1f" % (lo, hi), "Blocked by liveness": int(g.blocked.sum())})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\nFor attack scenarios (printed_photo, screen_replay, stranger) a LOW success % is good;"
          "\nfor genuine_trial a HIGH success % is good.")
    out = os.path.join(M.ATTENDANCE_DIR, "trial_summary.xlsx")
    res.to_excel(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()