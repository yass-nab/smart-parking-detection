"""
diagnose.py — run this FIRST to understand your video's pixel characteristics.
It samples a few slots and the lane areas, prints HSV stats, and saves a
side-by-side debug image so you can see exactly what's being sampled.

Usage:
    python diagnose.py --video parking_1920_1080.mp4 --slots slots.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


def extract_roi(frame, polygon):
    pts = np.array(polygon, dtype=np.float32)
    dst = np.array([[0,0],[79,0],[79,39],[0,39]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (80, 40))


def hsv_stats(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    return {
        "mean_H": float(np.mean(hsv[:,:,0])),
        "mean_S": float(np.mean(hsv[:,:,1])),
        "mean_V": float(np.mean(hsv[:,:,2])),
        "std_V":  float(np.std(hsv[:,:,2])),
        "edge%":  float(np.mean(edges > 0)) * 100,
        "dark%":  float(np.mean(hsv[:,:,2] < 60)) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--slots", required=True)
    ap.add_argument("--frame", type=int, default=0, help="Which frame to analyze (default: 0)")
    args = ap.parse_args()

    with open(args.slots) as f:
        slots = json.load(f)

    cap = cv2.VideoCapture(args.video)
    # seek to requested frame
    for _ in range(args.frame + 1):
        ret, frame = cap.read()
        if not ret:
            print("ERROR: could not read frame")
            return
    cap.release()

    H, W = frame.shape[:2]
    print(f"\nVideo frame size: {W}x{H}")
    print(f"Total slots: {len(slots)}")
    print(f"Slot x range: {min(p[0] for s in slots for p in s['polygon'])} – "
          f"{max(p[0] for s in slots for p in s['polygon'])}")
    print(f"Slot y range: {min(p[1] for s in slots for p in s['polygon'])} – "
          f"{max(p[1] for s in slots for p in s['polygon'])}")

    # ── Sample every 20th slot and print its stats ──────────────────────────
    print("\n── Slot pixel stats (every 20th slot) ──────────────────────────────")
    print(f"{'Slot':>5}  {'mean_H':>6} {'mean_S':>6} {'mean_V':>6} {'std_V':>5} {'edge%':>6} {'dark%':>6}")
    sample_rois = []
    for s in slots[::20]:
        roi = extract_roi(frame, s["polygon"])
        st  = hsv_stats(roi)
        print(f"  #{s['id']:3d}   {st['mean_H']:6.1f} {st['mean_S']:6.1f} "
              f"{st['mean_V']:6.1f} {st['std_V']:5.1f} {st['edge%']:6.1f} {st['dark%']:6.1f}")
        sample_rois.append((s["id"], roi, st))

    # ── Find x-gaps between parking columns (candidate lane positions) ───────
    col_ranges = []
    for s in slots:
        xs = [p[0] for p in s["polygon"]]
        col_ranges.append((min(xs), max(xs)))
    col_ranges.sort()
    merged = [list(col_ranges[0])]
    for x1, x2 in col_ranges[1:]:
        if x1 <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], x2)
        else:
            merged.append([x1, x2])

    print("\n── Detected parking columns (merged x-ranges) ──────────────────────")
    for c in merged:
        print(f"  x={c[0]}–{c[1]}  (width={c[1]-c[0]}px)")

    gaps = []
    for i in range(len(merged)-1):
        g0, g1 = merged[i][1]+2, merged[i+1][0]-2
        if g1 - g0 >= 6:
            gaps.append((g0, g1))

    print("\n── Gap regions between columns (driving lanes) ─────────────────────")
    if gaps:
        for g in gaps:
            print(f"  x={g[0]}–{g[1]}  (width={g[1]-g[0]}px)")
    else:
        print("  ⚠  No gaps found — slots may be touching or overlapping columns.")

    # ── Sample lane stats ───────────────────────────────────────────────────
    all_ys = [p[1] for s in slots for p in s["polygon"]]
    y0 = int(min(all_ys) + (max(all_ys) - min(all_ys)) * 0.2)
    y1 = int(max(all_ys) - (max(all_ys) - min(all_ys)) * 0.2)

    print(f"\n── Lane pixel stats (y={y0}–{y1}) ──────────────────────────────────")
    print(f"{'Lane':>10}  {'mean_H':>6} {'mean_S':>6} {'mean_V':>6} {'std_V':>5} {'edge%':>6}")
    lane_stats_list = []
    for g in gaps[:6]:
        roi = frame[y0:y1, g[0]:g[1]]
        if roi.size == 0:
            continue
        st = hsv_stats(roi)
        print(f"  x={g[0]:4d}-{g[1]:4d}  {st['mean_H']:6.1f} {st['mean_S']:6.1f} "
              f"{st['mean_V']:6.1f} {st['std_V']:5.1f} {st['edge%']:6.1f}")
        lane_stats_list.append(st)

    # ── Recommend thresholds ─────────────────────────────────────────────────
    if lane_stats_list:
        avg_lane_V   = np.mean([s["mean_V"]  for s in lane_stats_list])
        avg_lane_S   = np.mean([s["mean_S"]  for s in lane_stats_list])
        avg_lane_std = np.mean([s["std_V"]   for s in lane_stats_list])
        avg_lane_e   = np.mean([s["edge%"]   for s in lane_stats_list])
        slot_Vs = [hsv_stats(extract_roi(frame, s["polygon"]))["mean_V"] for s in slots[::5]]
        print(f"\n── Summary ──────────────────────────────────────────────────────────")
        print(f"  Lane avg brightness (V): {avg_lane_V:.1f}")
        print(f"  Lane avg saturation (S): {avg_lane_S:.1f}")
        print(f"  Lane avg edge density:   {avg_lane_e:.2f}%")
        print(f"  Slot V range (every 5th): min={min(slot_Vs):.1f}  max={max(slot_Vs):.1f}  "
              f"mean={np.mean(slot_Vs):.1f}")
        v_diff = abs(np.mean(slot_Vs) - avg_lane_V)
        print(f"  Avg slot vs lane V diff: {v_diff:.1f}  "
              f"({'significant — V-diff should help detect' if v_diff > 15 else 'small — V-diff alone wont work'  })")

    # ── Save annotated debug image ───────────────────────────────────────────
    debug = frame.copy()
    # Draw every slot outline in yellow
    for s in slots:
        poly = np.array(s["polygon"], dtype=np.int32)
        cv2.polylines(debug, [poly], True, (0, 220, 220), 1)
    # Draw lane bands in cyan
    for g in gaps:
        cv2.rectangle(debug, (g[0], y0), (g[1], y1), (255, 200, 0), 2)

    out = Path("output")
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "debug_frame.png"), debug)
    print(f"\n  Saved debug_frame.png → output/debug_frame.png")
    print("  Yellow = slot outlines,  Cyan = sampled lane bands")
    print("\nDone. Share the numbers above if you need further help tuning.\n")


if __name__ == "__main__":
    main()
