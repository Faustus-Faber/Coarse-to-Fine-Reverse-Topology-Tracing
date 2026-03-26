import cv2
import numpy as np
import math


def generate_gaming_video(output_path, fps=30.0, duration=2.0, size=(640, 480)):
    print("Generating simulated gaming test video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    frames = int(fps * duration)
    w, h = size

    # Pre-generate a massive 2000x2000 "world texture" map
    world_size = 2000
    world = np.zeros((world_size, world_size, 3), dtype=np.uint8)
    for y in range(0, world_size, 50):
        for x in range(0, world_size, 50):
            # Chessboard-like brick texture
            c = (40, 60, 40) if ((x//50) + (y//50)) % 2 == 0 else (60, 80, 60)
            cv2.rectangle(world, (x, y), (x+50, y+50), c, -1)
            # Add some 'terrain' details/dots
            cv2.circle(world, (x+25, y+25), 5, (100, 150, 100), -1)

    for i in range(frames):
        # 1. Simulate fast, non-linear mouse aim (camera pan over world)
        # Mouse aim is often quadratic or sinusoidal curve rather than linear
        cam_x = int(500 + i * 15 + math.sin(i * 0.2) * 50)
        cam_y = int(500 + i * 5 + math.cos(i * 0.1) * 30)

        # Extract the camera view from the world texture
        if cam_y + h <= world_size and cam_x + w <= world_size:
            frame = world[cam_y:cam_y+h, cam_x:cam_x+w].copy()
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 2. Draw IN-WORLD moving object (e.g. Enemy Model)
        # It moves diagonally and scales up (simulating running towards player)
        enemy_w = int(40 + i * 2)
        enemy_h = int(80 + i * 4)
        enemy_x = int(w/2 - 100 + i * 8)
        enemy_y = int(h/2 - enemy_h/2)
        cv2.rectangle(frame, (enemy_x, enemy_y), (enemy_x +
                      enemy_w, enemy_y + enemy_h), (0, 0, 200), -1)
        # Enemy 'head'
        cv2.circle(frame, (enemy_x + enemy_w//2, enemy_y),
                   enemy_w//3, (0, 0, 255), -1)

        # 3. DRAW STATIC HUD (The ultimate optical flow destroyer)
        # The background is moving wildly under these static pixels!

        # Crosshair (Center)
        cx, cy = w//2, h//2
        cv2.line(frame, (cx-10, cy), (cx+10, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy-10), (cx, cy+10), (0, 255, 0), 2)

        # Minimap (Top Right)
        mm_x, mm_y, mm_s = w - 120, 20, 100
        cv2.rectangle(frame, (mm_x, mm_y),
                      (mm_x+mm_s, mm_y+mm_s), (30, 30, 30), -1)
        cv2.rectangle(frame, (mm_x, mm_y), (mm_x+mm_s,
                      mm_y+mm_s), (200, 200, 200), 2)
        # Minimap details (static relative to minimap UI)
        cv2.circle(frame, (mm_x + 50, mm_y + 50), 5, (0, 255, 0), -1)

        # Health Bar (Bottom Left)
        hb_x, hb_y = 20, h - 50
        cv2.rectangle(frame, (hb_x, hb_y),
                      (hb_x + 150, hb_y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (hb_x, hb_y),
                      (hb_x + 120, hb_y + 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (hb_x, hb_y), (hb_x + 150,
                      hb_y + 20), (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Generated gaming video: {output_path}")


if __name__ == '__main__':
    generate_gaming_video(
        'C:\\Users\\farha\\OneDrive\\Desktop\\Planning\\gaming_input_30fps.mp4')
