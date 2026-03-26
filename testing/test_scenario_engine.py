"""
Parametric Scenario Rendering Engine
Each scenario is a unique configuration driving fundamentally different game simulations.
"""
import cv2
import numpy as np
import os

W, H = 640, 480
BASE = r'C:\Users\farha\OneDrive\Desktop\Planning\dataset'

# ───────────────────────── DRAWING PRIMITIVES ─────────────────────────


def draw_checkerboard(frame, offset_x, offset_y, tile=40, color1=(200, 200, 200), color2=(50, 50, 50)):
    for y in range(0, H+tile, tile):
        for x in range(0, W+tile, tile):
            ry, rx = y - int(offset_y) % tile, x - int(offset_x) % tile
            if (((x+int(offset_x))//tile + (y+int(offset_y))//tile) % 2) == 0:
                cv2.rectangle(frame, (rx, ry), (rx+tile, ry+tile), color1, -1)
            else:
                cv2.rectangle(frame, (rx, ry), (rx+tile, ry+tile), color2, -1)


def draw_starfield(frame, stars_x, stars_y, stars_b, offset_y=0):
    for sx, sy, sb in zip(stars_x, stars_y, stars_b):
        py = int((sy + offset_y) % H)
        cv2.circle(frame, (int(sx), py), 1, (int(sb), int(sb), int(sb)), -1)


def draw_ground_plane(frame, offset_x, stripe_w=80, colors=((80, 120, 40), (60, 100, 30))):
    for x in range(0, W + stripe_w, stripe_w):
        rx = x - int(offset_x) % stripe_w
        c = colors[((x + int(offset_x)) // stripe_w) % 2]
        cv2.rectangle(frame, (rx, H//2), (rx+stripe_w, H), c, -1)


def draw_building(frame, x, y, w_b, h_b, color):
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w_b), int(y+h_b)), color, -1)
    # windows
    for wy in range(int(y)+8, int(y+h_b)-8, 20):
        for wx in range(int(x)+6, int(x+w_b)-6, 16):
            cv2.rectangle(frame, (wx, wy), (wx+8, wy+12), (200, 220, 255), -1)


def draw_crosshair(frame, cx, cy, size=15, color=(0, 255, 0), thickness=1):
    cv2.line(frame, (cx-size, cy), (cx+size, cy), color, thickness)
    cv2.line(frame, (cx, cy-size), (cx, cy+size), color, thickness)
    cv2.circle(frame, (cx, cy), size//2, color, thickness)


def draw_health_bar(frame, x, y, w_bar, pct, color=(0, 220, 0)):
    cv2.rectangle(frame, (x, y), (x+w_bar, y+14), (40, 40, 40), -1)
    cv2.rectangle(frame, (x+1, y+1), (x+1+int((w_bar-2)*pct), y+13), color, -1)
    cv2.rectangle(frame, (x, y), (x+w_bar, y+14), (200, 200, 200), 1)


def draw_minimap(frame, x, y, sz, dots, dot_col=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x+sz, y+sz), (20, 30, 20), -1)
    cv2.rectangle(frame, (x, y), (x+sz, y+sz), (0, 200, 0), 1)
    for dx, dy in dots:
        cv2.circle(frame, (x+int(dx*sz), y+int(dy*sz)), 3, dot_col, -1)


def draw_ammo(frame, x, y, current, total):
    cv2.rectangle(frame, (x, y), (x+90, y+30), (20, 20, 20), -1)
    cv2.putText(frame, f"{current}/{total}", (x+5, y+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_scope_overlay(frame, cx, cy, radius):
    mask = np.ones_like(frame) * 0
    cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)
    frame[mask[:, :, 0] == 0] = 0
    cv2.circle(frame, (cx, cy), radius, (30, 30, 30), 3)
    cv2.line(frame, (cx-radius, cy), (cx+radius, cy), (30, 30, 30), 1)
    cv2.line(frame, (cx, cy-radius), (cx, cy+radius), (30, 30, 30), 1)


def draw_polygon_ship(frame, cx, cy, size, angle, color):
    pts = []
    for k, r in enumerate([size, size*0.6, size, size*0.6, size]):
        a = angle + k * 72
        pts.append([int(cx + r*np.cos(np.radians(a))),
                   int(cy + r*np.sin(np.radians(a)))])
    cv2.fillPoly(frame, [np.array(pts)], color)


def draw_car(frame, cx, cy, w_c, h_c, color):
    cv2.rectangle(frame, (int(cx-w_c//2), int(cy-h_c//2)),
                  (int(cx+w_c//2), int(cy+h_c//2)), color, -1)
    # wheels
    for dx in [-w_c//3, w_c//3]:
        for dy in [-h_c//2, h_c//2]:
            cv2.circle(frame, (int(cx+dx), int(cy+dy)), 6, (30, 30, 30), -1)


def draw_explosion(frame, cx, cy, radius, intensity):
    for r in range(int(radius), 0, -3):
        alpha = intensity * (r / radius)
        c = (int(50*alpha), int(100*alpha), int(255*alpha))
        cv2.circle(frame, (int(cx), int(cy)), r, c, 2)

# ───────────────────────── SCENARIO GENERATOR ─────────────────────────


def generate_scenario(sid, name, render_fn, fps=60.0, duration=2.0):
    """Generate a single scenario's GT (60fps) and input (30fps) videos."""
    gt_path = os.path.join(BASE, 'gt', f's{sid:03d}_{name}_60fps.mp4')
    in_path = os.path.join(BASE, 'input', f's{sid:03d}_{name}_30fps.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_gt = cv2.VideoWriter(gt_path, fourcc, fps, (W, H))
    out_in = cv2.VideoWriter(in_path, fourcc, 30.0, (W, H))
    frames = int(fps * duration)
    for i in range(frames):
        t = i / frames  # normalized time 0..1
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        render_fn(frame, i, t, frames)
        out_gt.write(frame)
        if i % 2 == 0:
            out_in.write(frame)
    out_gt.release()
    out_in.release()
    print(f"  [{sid:03d}] {name}")
