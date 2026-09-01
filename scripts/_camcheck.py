"""Verify the Camo feed is live (not a placeholder) and MediaPipe detects a person."""
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

MODEL = Path("data/.cache/pose_landmarker_lite.task")

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    raise SystemExit("cannot open camera 0")

frames = []
t0 = time.time()
for _ in range(45):
    ok, f = cap.read()
    if ok and f is not None:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
elapsed = time.time() - t0
cap.release()

print(f"captured {len(frames)} frames in {elapsed:.1f}s "
      f"({len(frames)/elapsed:.1f} fps)")

diffs = [float(np.mean(np.abs(frames[i + 1].astype(np.int16)
                              - frames[i].astype(np.int16))))
         for i in range(len(frames) - 1)]
mean_d = sum(diffs) / len(diffs)
print(f"consecutive-frame pixel delta: mean {mean_d:.3f}  "
      f"max {max(diffs):.3f}  min {min(diffs):.3f}")
if mean_d < 0.05:
    print("  >> STATIC IMAGE - this is a Camo placeholder, not a live feed")
else:
    print("  >> LIVE FEED confirmed (pixels are changing)")

print()
opts = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL)),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
)
det = PoseLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
hits = 0
tries = 12
vis_scores = []
for _ in range(tries):
    ok, f = cap.read()
    if not ok:
        continue
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = det.detect(mpi)
    if res.pose_landmarks:
        hits += 1
        lms = res.pose_landmarks[0]
        vis_scores.append(sum(l.visibility for l in lms) / len(lms))
    time.sleep(0.08)
cap.release()

print(f"MediaPipe person detection: {hits}/{tries} frames")
if vis_scores:
    print(f"  mean landmark visibility: {sum(vis_scores)/len(vis_scores):.3f}")
    print("  >> READY for ground-truth collection")
else:
    print("  >> NO PERSON DETECTED - make sure you are in frame, "
          "full body visible")
