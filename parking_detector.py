"""
Smart Parking Detection System  v6
====================================
Key insight from diagnostics:
  - Lane V mean (~121) ≈ Slot V mean (~122) → brightness is useless
  - Lane saturation ≈ Slot saturation → colour is useless
  - Lane edge% (~14%) ≈ Slot edge% → edges are unreliable

  The ONE signal that works: std_V (texture variance within the ROI)
    - Empty asphalt is UNIFORM  → low std_V  (~20-30)
    - Cars have panels/windows/shadows → high std_V (~40-60)

  Secondary signal: spatial contrast (difference between bright and dark
  sub-regions inside the slot). A car always has at least one dark window
  or tyre shadow next to a bright roof panel. Empty asphalt does not.

Usage:
    C:\Python313\python.exe parking_detector.py --video parking_1920_1080.mp4 --slots slots.json --frames 120
    C:\Python313\python.exe parking_detector.py --video parking_1920_1080.mp4 --slots slots.json --frames 120 --debug
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIER  — texture-variance based
# ─────────────────────────────────────────────────────────────────────────────

class SlotClassifier:
    """
    Uses local texture variance and spatial contrast rather than
    absolute colour/brightness — both of which are useless in this footage.

    Tuned from diagnostic output:
      Occupied slots: std_V ~ 40-55, high spatial contrast
      Empty slots:    std_V ~ 21-28, low spatial contrast

    OCCUPIED_THRESH / EMPTY_THRESH can be tuned via --debug output.
    """

    OCCUPIED_THRESH = 0.42   # score above this → OCCUPIED
    EMPTY_THRESH    = 0.22   # score below this → EMPTY

    def classify(self, roi: np.ndarray, debug: bool = False) -> tuple[str, float, float]:
        if roi is None or roi.size == 0:
            return "EMPTY", 0.5, 0.0
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return "EMPTY", 0.5, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        v    = hsv[:, :, 2]

        # ── Feature 1: global texture variance ─────────────────────────────
        # Cars: high std_V.  Empty asphalt: low std_V.
        std_v = float(np.std(v))
        # Normalise: 0 = perfectly uniform, 1 = very high variance
        # Range from diagnostics: ~20 (empty) to ~55 (occupied)
        f1 = min(std_v / 55.0, 1.0)

        # ── Feature 2: spatial contrast (quad analysis) ─────────────────────
        # Divide ROI into 4 quadrants; measure max-min of quad means.
        # A car always has light roof + dark window/tyre somewhere.
        # Empty asphalt has uniform quads.
        mid_h, mid_w = h // 2, w // 2
        quads = [
            v[:mid_h, :mid_w],
            v[:mid_h, mid_w:],
            v[mid_h:, :mid_w],
            v[mid_h:, mid_w:],
        ]
        quad_means = [float(np.mean(q)) for q in quads if q.size > 0]
        spatial_contrast = max(quad_means) - min(quad_means)
        # Normalise: 0 = flat, 1 = very high contrast
        # Empty asphalt: ~5-15.  Car with window: ~40-80.
        f2 = min(spatial_contrast / 100.0, 1.0)

        # ── Feature 3: dark-patch presence ─────────────────────────────────
        # Car windows / tyres appear as coherent dark blobs.
        # Use a lower threshold than v4 (60→80) to catch more cars.
        dark_frac = float(np.mean(v < 80))
        f3 = min(dark_frac * 4.0, 1.0)

        # ── Feature 4: local gradient energy ───────────────────────────────
        # Sobel captures fine texture (panel edges, window frames).
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        # Empty asphalt: ~5-15.  Car surface: ~20-50.
        f4 = min(grad_energy / 120.0, 1.0)

        # ── Feature 5: bright-spot ratio ───────────────────────────────────
        # Car roofs are often brighter than surrounding asphalt (specular).
        bright_frac = float(np.mean(v > 180))
        f5 = min(bright_frac * 5.0, 1.0)

        # ── Weighted score ──────────────────────────────────────────────────
        score = (
            0.38 * f1 +   # texture variance   — strongest signal (boosted)
            0.25 * f2 +   # spatial contrast
            0.15 * f3 +   # dark patches
            0.12 * f4 +   # gradient energy (reduced — was saturating at 1.0)
            0.10 * f5     # bright spots
        )

        if debug:
            print(f"    f1(std_v)={f1:.3f}  f2(contrast)={f2:.3f}  "
                  f"f3(dark)={f3:.3f}  f4(grad)={f4:.3f}  f5(bright)={f5:.3f}"
                  f"  → score={score:.3f}  std_v={std_v:.1f}  contrast={spatial_contrast:.1f}")

        if score >= self.OCCUPIED_THRESH:
            conf = min(0.50 + (score - self.OCCUPIED_THRESH) /
                       (1.0 - self.OCCUPIED_THRESH + 1e-6) * 0.45, 0.99)
            return "OCCUPIED", conf, score
        elif score <= self.EMPTY_THRESH:
            conf = min(0.50 + (self.EMPTY_THRESH - score) /
                       (self.EMPTY_THRESH + 1e-6) * 0.45, 0.99)
            return "EMPTY", conf, score
        else:
            span = self.OCCUPIED_THRESH - self.EMPTY_THRESH
            mid  = self.EMPTY_THRESH + span / 2
            conf = max(0.35, 0.50 - abs(score - mid) / span * 0.15)
            return "TRANSITION", conf, score


# ─────────────────────────────────────────────────────────────────────────────
#  OVERLAY RENDERER
# ─────────────────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "OCCUPIED":   (0,   0,   210),
    "EMPTY":      (0,   210,  50),
    "TRANSITION": (0,   200, 220),
}


def draw_slot(frame, slot, status, conf):
    poly  = np.array(slot["polygon"], dtype=np.int32)
    color = STATUS_COLORS[status]
    mask  = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    overlay = frame.copy()
    overlay[mask == 255] = color
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    cv2.polylines(frame, [poly], True, color, 1)
    cx = int(np.mean(poly[:, 0]))
    cy = int(np.mean(poly[:, 1]))
    cv2.putText(frame, f"#{slot['id']}", (cx - 10, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 255), 1, cv2.LINE_AA)
    bx = poly[:, 0].min()
    by = poly[:, 1].max() - 3
    bw = int((poly[:, 0].max() - bx) * conf)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 3), color, -1)


def draw_stats(frame, counts, frame_idx, fps):
    occ   = counts.get("OCCUPIED",   0)
    emp   = counts.get("EMPTY",      0)
    trans = counts.get("TRANSITION", 0)
    total = occ + emp + trans
    lines = [
        "PARKING ANALYSIS",
        f"  Empty      : {emp:4d}  ({100*emp/max(total,1):4.1f}%)",
        f"  Occupied   : {occ:4d}  ({100*occ/max(total,1):4.1f}%)",
        f"  Transition : {trans:4d}",
        f"  Total      : {total:4d}",
        f"  Frame      : {frame_idx:4d}  ({frame_idx/max(fps,1):.1f}s)",
    ]
    pad, lh = 8, 22
    box_h = len(lines) * lh + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (285, 5 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for i, line in enumerate(lines):
        color = (200, 200, 200)
        if "Empty"  in line: color = STATUS_COLORS["EMPTY"]
        elif "Occup" in line: color = STATUS_COLORS["OCCUPIED"]
        elif "PARK"  in line: color = (255, 255, 255)
        cv2.putText(frame, line, (12, pad + (i+1)*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  SLOT ANNOTATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_annotator(video_path, slots_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: cannot read first frame.")
        return
    slots, current_pts = [], []
    clone = frame.copy()
    win = "Annotator | Click 4 corners clockwise → SPACE confirm | Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 255), -1)
            if len(current_pts) > 1:
                cv2.line(clone, current_pts[-2], current_pts[-1], (0, 255, 255), 2)
            cv2.imshow(win, clone)

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, clone)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' ') and len(current_pts) == 4:
            sid = len(slots) + 1
            slots.append({"id": sid,
                          "polygon": [[int(p[0]), int(p[1])] for p in current_pts]})
            cv2.polylines(clone, [np.array(current_pts, np.int32)], True, (0,255,0), 2)
            cv2.putText(clone, str(sid), current_pts[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            current_pts.clear()
            print(f"  Slot {sid} saved.")
            cv2.imshow(win, clone)
    cv2.destroyAllWindows()
    with open(slots_path, "w") as f:
        json.dump(slots, f, indent=2)
    print(f"Saved {len(slots)} slots → {slots_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  ROI EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi(frame, polygon):
    pts = np.array(polygon, dtype=np.float32)
    dst = np.array([[0,0],[79,0],[79,39],[0,39]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (80, 40))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(video_path, slots_path, max_frames, out_gif, out_mp4, debug=False):
    with open(slots_path) as f:
        slots = json.load(f)
    print(f"Loaded {len(slots)} slots from {slots_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} @ {fps:.1f} fps  ({total} frames total)")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    mp4_path = str(out_dir / out_mp4)
    gif_path = str(out_dir / out_gif)

    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    classifier  = SlotClassifier()
    gif_frames  = []
    slot_status = {s["id"]: ("EMPTY", 0.5) for s in slots}
    frame_idx   = 0
    processed   = 0
    SKIP        = 2

    print(f"Processing up to {max_frames} frames (every {SKIP}nd)...\n")

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SKIP != 0:
            continue
        processed += 1

        annotated = frame.copy()
        counts = {"OCCUPIED": 0, "EMPTY": 0, "TRANSITION": 0}

        for slot in slots:
            roi = extract_roi(frame, slot["polygon"])
            do_debug = debug and processed == 1
            status, conf, score = classifier.classify(roi, debug=do_debug)
            if do_debug:
                print(f"  Slot {slot['id']:3d}: {status:<10} conf={conf:.2f}  score={score:.3f}")

            # Light temporal smoothing — only block very low-confidence flips
            prev_s, prev_c = slot_status[slot["id"]]
            if conf < 0.30 and prev_c >= 0.50:
                status, conf = prev_s, prev_c * 0.80
            slot_status[slot["id"]] = (status, conf)

            counts[status] += 1
            draw_slot(annotated, slot, status, conf)

        draw_stats(annotated, counts, frame_idx, fps)
        writer.write(annotated)

        if processed % 3 == 0:
            small = cv2.resize(annotated, (960, 540))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        if processed % 10 == 0:
            print(f"  Frame {frame_idx:4d} | "
                  f"Occ:{counts['OCCUPIED']:3d}  "
                  f"Emp:{counts['EMPTY']:3d}  "
                  f"Trans:{counts['TRANSITION']:3d}")

    cap.release()
    writer.release()
    print(f"\nMP4 → {mp4_path}")

    if gif_frames:
        try:
            import imageio
            imageio.mimsave(gif_path, gif_frames, fps=max(fps / SKIP / 3, 4))
            print(f"GIF → {gif_path}")
        except ImportError:
            print("imageio not installed — GIF skipped")

    print(f"\nDone. Processed {processed} frames, {len(slots)} slots.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Smart Parking Detector v6")
    ap.add_argument("--video",    default="parking_1920_1080.mp4")
    ap.add_argument("--slots",    default="slots.json")
    ap.add_argument("--frames",   type=int, default=120)
    ap.add_argument("--out-gif",  default="parking_detection.gif")
    ap.add_argument("--out-mp4",  default="parking_detection.mp4")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--debug",    action="store_true",
                    help="Print per-slot scores on frame 1 for threshold tuning")
    args = ap.parse_args()

    if args.annotate:
        run_annotator(args.video, args.slots)
    else:
        for path, name in [(args.video, "video"), (args.slots, "slots")]:
            if not Path(path).exists():
                print(f"ERROR: {name} file not found: {path}")
                sys.exit(1)
        run_detection(args.video, args.slots, args.frames,
                      args.out_gif, args.out_mp4, debug=args.debug)


if __name__ == "__main__":
    main()
