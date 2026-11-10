"""
Fast Parking Slot Annotator
============================
Click 4 corners per slot → SPACE to confirm → Q to quit and save.

Speed features:
  - Shows slot count and last saved slot live
  - CTRL+Z to undo last slot
  - S to save progress without quitting (auto-saves every 10 slots too)
  - Zoom in/out with mouse wheel
  - Existing slots.json loaded and shown in green so you don't re-do done ones
  - Output saved to slots_new.json (won't overwrite your original)

Usage:
    C:\Python313\python.exe annotator.py --video parking_1920_1080.mp4
    C:\Python313\python.exe annotator.py --video parking_1920_1080.mp4 --resume slots_new.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--resume", default="slots_new.json",
                    help="Load existing annotations to continue from where you left off")
    ap.add_argument("--out",    default="slots_new.json")
    args = ap.parse_args()

    # ── Load first frame ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    ret, original_frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: cannot read video frame")
        return

    H, W = original_frame.shape[:2]

    # ── Load existing slots if resuming ──────────────────────────────────────
    slots = []
    if Path(args.resume).exists():
        with open(args.resume) as f:
            slots = json.load(f)
        print(f"Loaded {len(slots)} existing slots from {args.resume}")

    current_pts = []
    zoom        = 1.0
    pan_x       = 0
    pan_y       = 0
    dragging    = False
    drag_start  = (0, 0)
    pan_start   = (0, 0)

    WIN = "Annotator  |  Click 4 corners  |  SPACE=confirm  CTRL+Z=undo  S=save  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(W, 1400), min(H, 800))

    def to_frame_coords(sx, sy):
        """Convert screen click → original frame coordinates."""
        fx = int(sx / zoom + pan_x)
        fy = int(sy / zoom + pan_y)
        return max(0, min(fx, W-1)), max(0, min(fy, H-1))

    def to_screen_coords(fx, fy):
        """Convert frame coords → screen coords for drawing."""
        sx = int((fx - pan_x) * zoom)
        sy = int((fy - pan_y) * zoom)
        return sx, sy

    def render():
        """Build the display image from scratch each frame."""
        # Crop and zoom the relevant portion
        view_w = int(W / zoom)
        view_h = int(H / zoom)
        x1 = max(0, pan_x)
        y1 = max(0, pan_y)
        x2 = min(W, pan_x + view_w)
        y2 = min(H, pan_y + view_h)
        crop = original_frame[y1:y2, x1:x2].copy()

        # Scale to window size
        disp_w = int((x2 - x1) * zoom)
        disp_h = int((y2 - y1) * zoom)
        if disp_w < 1 or disp_h < 1:
            return original_frame.copy()
        canvas = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Draw confirmed slots in green
        for s in slots:
            poly_s = np.array([to_screen_coords(p[0], p[1]) for p in s["polygon"]], np.int32)
            cv2.polylines(canvas, [poly_s], True, (0, 220, 60), 1)
            cx = int(np.mean(poly_s[:, 0]))
            cy = int(np.mean(poly_s[:, 1]))
            cv2.putText(canvas, str(s["id"]), (cx-8, cy+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 80), 1)

        # Draw current in-progress points in yellow
        for i, (fx, fy) in enumerate(current_pts):
            sx, sy = to_screen_coords(fx, fy)
            cv2.circle(canvas, (sx, sy), 5, (0, 220, 220), -1)
            if i > 0:
                px, py = to_screen_coords(current_pts[i-1][0], current_pts[i-1][1])
                cv2.line(canvas, (px, py), (sx, sy), (0, 220, 220), 1)

        # HUD
        lines = [
            f"Slots saved: {len(slots)}",
            f"Points this slot: {len(current_pts)}/4",
            f"Zoom: {zoom:.1f}x",
            "SPACE=confirm  CTRL+Z=undo  S=save  Q=quit",
            "Scroll=zoom  Middle-drag=pan",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)

    def on_mouse(event, sx, sy, flags, _):
        nonlocal zoom, pan_x, pan_y, dragging, drag_start, pan_start

        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = to_frame_coords(sx, sy)
            current_pts.append((fx, fy))
            render()

        elif event == cv2.EVENT_MBUTTONDOWN:
            dragging   = True
            drag_start = (sx, sy)
            pan_start  = (pan_x, pan_y)

        elif event == cv2.EVENT_MBUTTONUP:
            dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = int((drag_start[0] - sx) / zoom)
            dy = int((drag_start[1] - sy) / zoom)
            pan_x = max(0, min(pan_start[0] + dx, W - int(W/zoom)))
            pan_y = max(0, min(pan_start[1] + dy, H - int(H/zoom)))
            render()

        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = zoom
            if flags > 0:
                zoom = min(zoom * 1.15, 10.0)
            else:
                zoom = max(zoom / 1.15, 0.3)
            # Zoom toward mouse position
            fx, fy = to_frame_coords(sx, sy)
            pan_x = int(fx - sx / zoom)
            pan_y = int(fy - sy / zoom)
            pan_x = max(0, min(pan_x, W - int(W/zoom)))
            pan_y = max(0, min(pan_y, H - int(H/zoom)))
            render()

    cv2.setMouseCallback(WIN, on_mouse)
    render()

    def save():
        with open(args.out, "w") as f:
            json.dump(slots, f, indent=2)
        print(f"  Saved {len(slots)} slots → {args.out}")

    print("\nControls:")
    print("  Left-click    : place corner point (need 4)")
    print("  SPACE         : confirm slot (after 4 points)")
    print("  CTRL+Z        : undo last slot")
    print("  S             : save progress")
    print("  Q / ESC       : save and quit")
    print("  Scroll wheel  : zoom in/out")
    print("  Middle-drag   : pan\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        ctrl = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1

        if key == ord('q') or key == 27:
            save()
            break

        elif key == ord('s'):
            save()

        elif key == ord(' '):
            if len(current_pts) == 4:
                sid = len(slots) + 1
                slots.append({"id": sid,
                               "polygon": [[p[0], p[1]] for p in current_pts]})
                current_pts.clear()
                print(f"  Slot {sid} confirmed  (total: {len(slots)})")
                if len(slots) % 10 == 0:
                    save()
                render()
            else:
                print(f"  Need 4 points, have {len(current_pts)}")

        elif key == 26:  # CTRL+Z
            if slots:
                removed = slots.pop()
                print(f"  Undid slot {removed['id']}")
                render()
            elif current_pts:
                current_pts.pop()
                print(f"  Removed last point")
                render()

    cv2.destroyAllWindows()
    print(f"\nDone. {len(slots)} slots saved to {args.out}")
    print(f"Use --slots {args.out} when running parking_detector.py")


if __name__ == "__main__":
    main()
